"""
QM9star PostgreSQL dump → xyz 변환 전처리 스크립트.

PostgreSQL binary dump 파일에서 snapshot/molecule 데이터를 추출하여
ASE가 읽을 수 있는 extended .xyz 파일로 변환합니다.

사용법:
  1) pg_restore 사용 가능한 경우 (권장):
     python qm9star_preprocessor.py --dump qm9star_archive_240912.sql --method pg_restore

  2) pg_restore 없이 plain SQL 파일이 이미 있는 경우:
     python qm9star_preprocessor.py --sql qm9star_plain.sql --method parse_sql

단위 변환:
  - 에너지: Hartree → eV (×27.2114)
  - 힘: Hartree/bohr → eV/Å (×51.4221)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import numpy as np
from pathlib import Path


# 단위 변환 상수
HARTREE_TO_EV = 27.211386245988  # eV/Hartree
BOHR_TO_ANGSTROM = 0.529177249   # Å/bohr
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM  # eV/Å per Hartree/bohr

# 원소 기호 → 원자번호 매핑
ELEMENT_SYMBOLS = [
    'X',  # 0
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
]


def dump_to_plain_sql(dump_path, output_sql_path):
    """pg_restore를 사용하여 binary dump를 plain SQL로 변환"""
    print(f"[1/3] pg_restore로 plain SQL 변환 중...")
    print(f"  입력: {dump_path}")
    print(f"  출력: {output_sql_path}")

    try:
        result = subprocess.run(
            ['pg_restore', '-f', str(output_sql_path), str(dump_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # pg_restore는 DB 연결 없이도 -f 옵션으로 출력 가능
            # 일부 경고는 무시
            if "database" in result.stderr.lower():
                print(f"  경고 (무시 가능): {result.stderr[:200]}")
            else:
                print(f"  에러: {result.stderr[:500]}")
                return False
        print(f"  완료!")
        return True
    except FileNotFoundError:
        print("  ❌ pg_restore를 찾을 수 없습니다.")
        print("  설치: sudo apt install postgresql-client")
        return False


def parse_pg_array(s):
    """PostgreSQL 배열 문자열을 Python list로 변환.
    예: '{1,2,3}' → [1, 2, 3]
        '{{1,2,3},{4,5,6}}' → [[1,2,3],[4,5,6]]
    """
    if s is None or s == '\\N' or s == '':
        return None

    s = s.strip()
    if not s.startswith('{'):
        return None

    # 중첩 배열 처리
    depth = 0
    for c in s:
        if c == '{':
            depth += 1
        else:
            break

    if depth == 1:
        # 1D 배열
        inner = s[1:-1]
        if not inner:
            return []
        return [float(x) if '.' in x or 'e' in x.lower()
                else int(x)
                for x in inner.split(',')]
    elif depth == 2:
        # 2D 배열 {{1,2,3},{4,5,6}}
        inner = s[1:-1]  # {1,2,3},{4,5,6}
        rows = []
        current_row = []
        in_row = False
        buf = ""
        for c in inner:
            if c == '{':
                in_row = True
                buf = ""
            elif c == '}':
                if buf:
                    current_row.append(
                        float(buf) if '.' in buf or 'e' in buf.lower()
                        else int(buf)
                    )
                rows.append(current_row)
                current_row = []
                in_row = False
                buf = ""
            elif c == ',':
                if in_row:
                    if buf:
                        current_row.append(
                            float(buf) if '.' in buf or 'e' in buf.lower()
                            else int(buf)
                        )
                    buf = ""
                # 행 사이의 쉼표는 무시
            else:
                buf += c
        return rows
    return None


def parse_copy_data(lines, table_name, columns):
    """
    COPY 구문의 데이터 행들을 파싱.

    Args:
        lines: SQL 파일에서 COPY 구문 이후의 데이터 라인들
        table_name: 테이블 이름
        columns: 컬럼 이름 리스트

    Returns:
        list of dict, 각 dict는 {column_name: value}
    """
    records = []
    for line in lines:
        line = line.rstrip('\n')
        if line == '\\.':
            break
        values = line.split('\t')
        record = {}
        for col, val in zip(columns, values):
            if val == '\\N':
                record[col] = None
            else:
                record[col] = val
        records.append(record)
    return records


def parse_sql_file(sql_path, charge_type="npa_charges",
                   energy_type="U_0", max_samples=None):
    """
    Plain SQL 파일에서 snapshot + molecule 데이터를 읽기.

    Args:
        sql_path: plain SQL 파일 경로
        charge_type: 사용할 charge 종류
        energy_type: 사용할 에너지 종류
        max_samples: 최대 샘플 수 (None이면 전체)

    Returns:
        list of dict with keys:
        - atoms, coords, forces, energy
        - atomic_charges, total_charge
        - smiles, molecule_id
    """
    print(f"[2/3] SQL 파일 파싱 중: {sql_path}")
    print(f"  charge_type: {charge_type}")
    print(f"  energy_type: {energy_type}")

    # 먼저 molecule 테이블에서 total_charge 읽기
    molecules = {}  # molecule_id → total_charge

    snapshot_columns = []
    molecule_columns = []
    snapshots = []

    with open(sql_path, 'r', encoding='utf-8', errors='replace') as f:
        in_copy = False
        current_table = None
        current_columns = []
        data_lines = []

        for line in f:
            # COPY 구문 시작 감지
            if line.startswith('COPY '):
                match = re.match(
                    r'COPY\s+(?:public\.)?(\S+)\s+\((.+?)\)\s+FROM\s+stdin',
                    line
                )
                if match:
                    current_table = match.group(1).strip('"')
                    col_str = match.group(2)
                    current_columns = [
                        c.strip().strip('"')
                        for c in col_str.split(',')
                    ]
                    in_copy = True
                    data_lines = []
                    continue

            if in_copy:
                if line.rstrip('\n') == '\\.':
                    # COPY 데이터 끝
                    if current_table == 'molecule':
                        records = parse_copy_data(
                            data_lines, current_table, current_columns
                        )
                        for r in records:
                            mid = r.get('id')
                            tc = r.get('total_charge')
                            if mid and tc:
                                molecules[mid] = float(tc)
                        print(f"  molecule 테이블: {len(molecules)}개 로드")

                    elif current_table == 'snapshot':
                        records = parse_copy_data(
                            data_lines, current_table, current_columns
                        )
                        snapshots = records
                        print(f"  snapshot 테이블: {len(snapshots)}개 로드")

                    in_copy = False
                    current_table = None
                    continue

                data_lines.append(line)

    # 데이터 변환
    print(f"\n[3/3] 데이터 변환 중...")
    results = []
    skipped = 0

    for i, snap in enumerate(snapshots):
        if max_samples and len(results) >= max_samples:
            break

        try:
            # 원자 정보
            atoms = parse_pg_array(snap.get('atoms'))
            coords = parse_pg_array(snap.get('coords'))
            if atoms is None or coords is None:
                skipped += 1
                continue

            # 에너지
            energy_val = snap.get(energy_type)
            if energy_val is None or energy_val == '':
                energy_val = snap.get('single_point_energy')
            if energy_val is None:
                skipped += 1
                continue
            energy = float(energy_val) * HARTREE_TO_EV

            # 힘
            forces_raw = parse_pg_array(snap.get('forces'))
            if forces_raw is not None:
                forces = np.array(forces_raw) * HARTREE_BOHR_TO_EV_ANGSTROM
            else:
                forces = np.zeros((len(atoms), 3))

            # Charge
            charges_raw = parse_pg_array(snap.get(charge_type))
            if charges_raw is not None:
                atomic_charges = np.array(charges_raw, dtype=float)
            else:
                # fallback
                for ct in ['npa_charges', 'mulliken_charge',
                           'hirshfeld_charges', 'formal_charges']:
                    charges_raw = parse_pg_array(snap.get(ct))
                    if charges_raw is not None:
                        atomic_charges = np.array(charges_raw, dtype=float)
                        break
                else:
                    atomic_charges = np.zeros(len(atoms))

            # Total charge (molecule 테이블에서)
            mol_id = snap.get('molecule_id')
            total_charge = molecules.get(mol_id, 0.0)

            result = {
                'atoms': np.array(atoms, dtype=int),
                'coords': np.array(coords, dtype=float),
                'forces': forces.tolist() if isinstance(forces, np.ndarray) 
                          else forces,
                'energy': energy,
                'atomic_charges': atomic_charges.tolist() 
                    if isinstance(atomic_charges, np.ndarray) 
                    else atomic_charges,
                'total_charge': total_charge,
                'molecule_id': mol_id,
                'filename': snap.get('filename', ''),
            }
            results.append(result)

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  경고: record {i} 건너뜀 — {e}")

    print(f"  변환 완료: {len(results)}개 성공, {skipped}개 건너뜀")
    return results


def write_extended_xyz(results, output_path, max_per_file=None):
    """
    변환된 데이터를 extended xyz 형식으로 저장.

    ASE가 읽을 수 있는 extended xyz:
    - 첫 줄: 원자 수
    - 둘째 줄: Properties 정보 (Lattice, energy, pbc 등)
    - 나머지: element x y z fx fy fz charge
    """
    print(f"\n  xyz 파일 저장 중: {output_path}")

    with open(output_path, 'w') as f:
        for i, result in enumerate(results):
            if max_per_file and i >= max_per_file:
                break

            atoms = result['atoms']
            coords = result['coords']
            forces = result['forces']
            energy = result['energy']
            charges = result['atomic_charges']
            total_charge = result['total_charge']
            n_atoms = len(atoms)

            # Extended XYZ 첫 줄
            f.write(f"{n_atoms}\n")

            # Properties 라인
            lattice = "30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0"
            props = (
                f'Lattice="{lattice}" '
                f'Properties=species:S:1:pos:R:3:forces:R:3:charges:R:1 '
                f'energy={energy} '
                f'total_charge={total_charge} '
                f'pbc="F F F"'
            )
            f.write(f"{props}\n")

            # 원자 데이터
            for j in range(n_atoms):
                z = int(atoms[j])
                symbol = ELEMENT_SYMBOLS[z] if z < len(ELEMENT_SYMBOLS) else f"X{z}"
                x, y, zz = coords[j]
                if isinstance(forces, (list, np.ndarray)) and len(forces) > j:
                    fx, fy, fz = forces[j]
                else:
                    fx, fy, fz = 0.0, 0.0, 0.0
                q = charges[j] if j < len(charges) else 0.0
                f.write(
                    f"{symbol:2s} {x:16.8f} {y:16.8f} {zz:16.8f} "
                    f"{fx:16.8f} {fy:16.8f} {fz:16.8f} {q:12.6f}\n"
                )

    print(f"  {min(len(results), max_per_file or len(results))}개 구조 저장 완료")


def main():
    parser = argparse.ArgumentParser(
        description='QM9star PostgreSQL dump → extended XYZ 변환'
    )
    parser.add_argument(
        '--dump', type=str, default=None,
        help='PostgreSQL binary dump 파일 경로'
    )
    parser.add_argument(
        '--sql', type=str, default=None,
        help='Plain SQL 파일 경로 (pg_restore 이후)'
    )
    parser.add_argument(
        '--output', type=str, default='qm9star_data.xyz',
        help='출력 xyz 파일 경로'
    )
    parser.add_argument(
        '--charge-type', type=str, default='npa_charges',
        choices=['formal_charges', 'mulliken_charge', 'npa_charges',
                 'hirshfeld_charges', 'lowdin_charges'],
        help='사용할 charge 종류 (기본: npa_charges)'
    )
    parser.add_argument(
        '--energy-type', type=str, default='U_0',
        choices=['single_point_energy', 'U_0', 'U_T', 'H_T', 'G_T'],
        help='사용할 에너지 종류 (기본: U_0)'
    )
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='최대 샘플 수'
    )
    args = parser.parse_args()

    # Step 1: binary dump → plain SQL (필요시)
    sql_path = args.sql
    if sql_path is None and args.dump is not None:
        sql_path = args.dump.replace('.sql', '_plain.sql')
        if not os.path.exists(sql_path):
            success = dump_to_plain_sql(args.dump, sql_path)
            if not success:
                print("\n대안: pg_restore 없이 진행할 수 없습니다.")
                print("  sudo apt install postgresql-client-16")
                sys.exit(1)
        else:
            print(f"  이미 변환된 SQL 파일 사용: {sql_path}")

    if sql_path is None:
        print("--dump 또는 --sql 중 하나를 지정해주세요.")
        sys.exit(1)

    # Step 2: SQL 파싱
    results = parse_sql_file(
        sql_path,
        charge_type=args.charge_type,
        energy_type=args.energy_type,
        max_samples=args.max_samples,
    )

    if not results:
        print("데이터를 찾을 수 없습니다.")
        sys.exit(1)

    # Step 3: XYZ로 저장
    write_extended_xyz(results, args.output, args.max_samples)

    # 통계 출력
    energies = [r['energy'] for r in results]
    charges_flat = [q for r in results for q in r['atomic_charges']]
    print(f"\n=== 데이터 통계 ===")
    print(f"  총 구조 수: {len(results)}")
    print(f"  에너지 범위: {min(energies):.4f} ~ {max(energies):.4f} eV")
    print(f"  charge 범위: {min(charges_flat):.4f} ~ {max(charges_flat):.4f}")
    print(f"  출력 파일: {args.output}")


if __name__ == '__main__':
    main()
