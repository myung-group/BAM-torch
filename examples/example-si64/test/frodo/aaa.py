import torch

# TorchScript 모델 로드
model = torch.load("model.pt-lammps.pt", map_location="cpu")

# 모델 전체 구조 출력
print(model)
print(dir(model))

if hasattr(model, "r_max"):
    print(f"r_max: {model.r_max}")
else:
    print("r_max 속성이 존재하지 않습니다.")

if hasattr(model, "atomic_numbers"):
    print(f"atomic_numbers: {model.atomic_numbers}")
else:
    print("atomic_numbers 속성이 존재하지 않습니다.")

if hasattr(model, "num_interactions"):
    print(f"num_interactions: {model.num_interactions}")
else:
    print("num_interactions 속성이 존재하지 않습니다.")
