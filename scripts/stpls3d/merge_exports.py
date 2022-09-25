import os
import shutil
from glob import glob
from tqdm import tqdm

base_path = "INSERT_WORKING_DIRECTORY"
vs03 = f"{base_path}/benchmark_03"
vs02 = f"{base_path}/benchmark_02"

target_path = "INSERT_TARGET_DIRECTORY"

print("COPY MASKS FILES 1/2 ...")
shutil.copytree(f"{vs02}/pred_mask", f"{target_path}/pred_mask_02")
print("COPY MASKS FILES 2/2 ...")
shutil.copytree(f"{vs03}/pred_mask", f"{target_path}/pred_mask_03")

for scene03 in tqdm(glob(f"{vs03}/*.txt")):
    instances = []
    with open(scene03, "r") as file03:
        while line := file03.readline().rstrip():
            mask_path, class_id, score = line.split(" ")

            if int(class_id) in [1, 3, 4, 7, 8, 11, 12, 13]:
                instances.append(f'{mask_path.replace("pred_mask", "pred_mask_03")} {class_id} {score}')
                print(instances[-1])
            else:
                print(f'DELETE {target_path}/{mask_path.replace("pred_mask", "pred_mask_03")}')
                os.remove(f'{target_path}/{mask_path.replace("pred_mask", "pred_mask_03")}')

        with open(f'{vs02}/{scene03.split("/")[-1]}', "r") as file02:
            while line := file02.readline().rstrip():
                mask_path, class_id, score = line.split(" ")

                if int(class_id) not in [1, 3, 4, 7, 8, 11, 12, 13]:
                    instances.append(f'{mask_path.replace("pred_mask", "pred_mask_02")} {class_id} {score}')
                    print(instances[-1])
                else:
                    print(f'DELETE {target_path}/{mask_path.replace("pred_mask", "pred_mask_02")}')
                    os.remove(f'{target_path}/{mask_path.replace("pred_mask", "pred_mask_02")}')

    with open(f'{target_path}/{scene03.split("/")[-1]}', 'w') as fout:
        for line in instances:
            fout.write(f"{line}\n")
