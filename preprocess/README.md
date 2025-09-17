# Preprocess
We adopt [Sapiens](https://github.com/facebookresearch/sapiens/tree/main) to infer pose skeletons and hand normals from the driving video.

## Sapiens Models

(1) Download Sapiens checkpoints:

| Model | Original | TorchScript |
| :---  | :---:    | :---:       |
| sapiens-pose-bbox-detector |	[link](https://huggingface.co/facebook/sapiens-pose-bbox-detector/tree/main) | - |
| sapiens-pose-coco | - | [link](https://huggingface.co/noahcao/sapiens-pose-coco/tree/main/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b) |
| sapiens-seg       | - | [link](https://huggingface.co/facebook/sapiens-seg-1b-torchscript/tree/main) |
| sapiens-normal    | - | [link](https://huggingface.co/facebook/sapiens-normal-2b-torchscript/tree/main) |

(2) Arrange the file structure as follows:
```plaintext
sapiens
├── detector
│   └── rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
├── pose
│   └── sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2
├── normal
│   └── sapiens_2b_normal_render_people_epoch_70_torchscript.pt2
└── seg
    └── sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2
```

(3) Set the environment variable:
```bash
export SAPIENS_CKPT_ROOT=path/to/sapiens/checkpoints
```

## Sapiens Setup
(1) Clone Sapiens to your desired local path:
```bash
git clone https://github.com/facebookresearch/sapiens.git

export SAPIENS_REPO_ROOT=path/to/sapiens/repo
export POSEGEN_REPO_ROOT=path/to/posegen/repo
```

(2) Setup the conda environment:
```bash
conda create -n sapiens python=3.10 -y
conda activate sapiens

# pytorch >= 2.2 is needed
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# a function to install a package via pip with editable mode and verbose output
pip_install_editable() {
  echo "Installing $1..."
  cd "$1" || exit
  pip install -e . -v
  cd - || exit
  echo "Finished installing $1"
}

cd ${SAPIENS_REPO_ROOT}
pip_install_editable "engine"
pip_install_editable "cv"
pip install -r "cv/requirements/optional.txt"
pip_install_editable "pose"
pip_install_editable "det"
pip_install_editable "seg"

pip install decord
```

## Sapiens Inference
(1) Prepare inference scripts:
```bash
cp ${POSEGEN_REPO_ROOT}/preprocess/*.py ${SAPIENS_REPO_ROOT}/lite/demo/
```

(2) Run inference:
```bash
cd ${SAPIENS_REPO_ROOT}/lite

# pose estimation
python demo/inference_pose.py \
    --video_path "${POSEGEN_REPO_ROOT}/examples/video1.mp4" \
    --output_dir "${POSEGEN_REPO_ROOT}/results/video1/sapiens" \
    --batch_size 8

# surface normal estimation
python demo/inference_normal.py \
    --video_path "${POSEGEN_REPO_ROOT}/examples/video1.mp4" \
    --output_dir "${POSEGEN_REPO_ROOT}/results/video1/sapiens" \
    --batch_size 8

# body part segmentation
python demo/inference_seg.py \
    --video_path "${POSEGEN_REPO_ROOT}/examples/video1.mp4" \
    --output_dir "${POSEGEN_REPO_ROOT}/results/video1/sapiens" \
    --batch_size 8
```
