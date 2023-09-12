## Paint-by-Example Augmentation

### Data preparing
- https://drive.google.com/file/d/15QzaTWsvZonJcXsNv-ilMRCYaQLhzR_i/view 에서 pretrained pbe model 다운로드
- ./pretrained_models 로 다운로드 받은 모델 이동
- https://cnai1021-my.sharepoint.com/:f:/g/personal/geunuk_cnai_ai/Ekq958GwhMBNkHWGPLXN6HcBQlC_ZC-TYZJd98r-Kmt-2A?e=bUb28F 에서 dataset 다운로드
- 압축해제 후 ./ 로 이동
- 다운로드 받은 데이터 이동 후의 폴더 구조
```
Paint-by-Example
├── gookbang
│  ├── pbe_aug_res
│  ├── video
│  │  ├── A-15
│  │  │  ├── xxx.mp4
│  │  │  ├── xxx.txt
│  │  │  ├── . . .
│  │  ├── A-sunrise
│  │  ├── ...
│  │  ├── B-sunrise
│  ├── video2
│  │  ├── Aregion-09
│  │  │  ├── xxx.mp4
│  │  │  ├── xxx.txt
│  │  │  ├── . . .
│  │  ├── Aregion-12-1
│  │  ├── ...
│  │  ├── Aregion 15-2
```
### 실행
git clone https://github.com/cnaigithub/pbe_aug.git
cd Paint-by-Example
conda env create -f environment.yaml
conda activate Paint-by-Example
python pbe_aug.py 