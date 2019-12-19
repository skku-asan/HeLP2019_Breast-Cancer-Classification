# HeLP2019_Breast-Cancer-Classification

## 코드 설명
```
├── Dockfile              : docker image 생성을 위한 파일
└── src
    ├── get_major_axis.py : major axis 길이 구하는 코드
    ├── train.py          : 예시 train 코드
    ├── inference.py      : 예시 inference 코드
    ├── train.sh          : train.py 실행하는 shell script
    ├── inference.sh      : inference.py 실행하는 shell script
    ├── requirements.txt  : image 내에서 pip package install
    └── output.csv        : phase 1 testset list
    
```
- `train.py`, `inference.py`는 참고하여 수정
- `requirements.txt`는 docker container 내에서 `pip freeze > requirements.txt`로 생성


## 예제 코드 클라우드 내에서 실행하는 방법 (Ubuntu 18.04.2 LTS 기준)
- 이 repository를 **clone**한 뒤, Dockerfile이 있는 폴더로 위치하여 아래의 명령어 실행
```
> docker build --tag test:1 .             # docker build [OPTIONS] PATH | URL | -
> docker save test:1 | gzip > test.tar.gz # docker save [OPTIONS] IMAGE [IMAGE...]
```
- 클라우드 페이지에서 tasks 창 오른쪽 위에 **Upload**를 클릭
- **browse** 클릭하여 test.tar.gz 업로드
- 자동으로 train 및 inference가 진행
- **혹시 실행이 되지 않는다면 에러가 발생하는 부분을 수정해서 진행해주세요.**
