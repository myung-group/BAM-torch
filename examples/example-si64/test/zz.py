from ase.io import read, write

# .traj 파일에서 첫 번째 구조 읽기
atoms = read('train.traj', index=0)

# .xyz 형식으로 저장
write('train_si.xyz', atoms)

