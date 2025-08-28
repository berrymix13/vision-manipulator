# vision-manipulator

> **Vision + Robot Arm 통합 시스템 개발 프로젝트**  \
> YOLO 기반 객체 탐지, 키포인트 추출, 좌표 변환, Grasp 수행까지 이어지는 end-to-end 로봇 비전 자동화 시스템입니다.

<br>

## 프로젝트 개요

이 프로젝트는 **컴퓨터 비전 기반 로봇 팔 자동화 시스템**으로, 주요 기능은 다음과 같습니다:

- **YOLOv11 기반 객체 탐지**
- **LLM 기반 자연어 명령 해석**
- **Keypoint → 3D 좌표 변환**
- **로봇 팔 제어 (Eye-in-Hand 구조)**
- **RGB-D 이미지 캡처 및 Calibration**
- **기하 기반 Surface Normal 분석**

<br>

### 디렉토리 구조

```bash
vision-manipulator/
├── data/              # 캡처 이미지, Calibration Data,예제 영상
├── notebooks/         # 분석용 Jupyter 노트북
├── prompt/            # LLM 프롬프트 예시
├── utils/             # 보조 함수 모음 (좌표 변환 등)
├── main_executor.py   # 메인 실행 코드
├── README.md
└── .gitignore
```

<br>

## 주요 기능

| 기능                          | 설명                                        |
| --------------------------- | ----------------------------------------- |
| **Vision 기반 객체 인식**         | YOLOv11 fine-tuning 모델로 실시간 탐지            |
| **좌표 변환**                   | 2D keypoint → 3D 카메라 좌표 → 로봇 베이스 좌표       |
| **Eye-in-Hand Calibration** | 카메라→그리퍼 변환 행렬 계산 (`cv2.calibrateHandEye`) |
| **자연어 명령 처리**               | GPT 기반 LLM으로 자연어를 명령어로 해석                 |
| **Grasp 제어**                | 예측된 위치 기반 로봇팔 Pick & Place 동작 수행          |

<br>

## 환경 세팅
<table align="center" style="text-align: center;">
  <thead>
    <tr>
      <th><b>Eye‑to‑Hand Environment</b></th>
      <th><b>Calibration target(Camera View)</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img src="./data/asset/Environment4.jpg" height="300" align="center" />
      </td>
      <td>
        <img src="./data/asset/charuco_board.jpg" height="300" align="center" />
      </td>
    </tr>
  </tbody>
</table>

* **사용 하드웨어**
  * **Intel RealSense D455** \
    : Depth sensing 및 RGB 이미지 동시 수집을 위한 RGB-D 카메라. \
    : 카메라는 로봇의 외부에 Eye-to-Hand 방식으로 고정됨.
  * **Elephant Robotics MyCobot 280 M5** \
    : ROS2 호환 소형 로봇팔

* **통합 시스템 구성**
  * 카메라 → Depth + RGB 이미지 취득
  * ROI 추출 및 포인트클라우드 생성
  * Surface normal + PCA 기반 RPY 자세 추정
  * 추정된 6-DoF → 로봇 베이스 좌표계로 변환하여 pick pose 결정

* **기술 스택**
  * OS: Ubuntu 22.04
  * ROS: ROS2 Humble
  * Vision: OpenCV, Open3D
  * Detection: YOLOv11 기반 커스텀 객체 탐지


<br>

## 2. Hand-Eye Calibration 오차 분석
캘리브레이션 정밀도를 높이기 위해 **Tsai-Lenz 알고리즘 기반 변환 행렬**에 다양한 후처리 방식을 적용하고,  
XYZ 오차와 RPY 오차로 나누어 분석을 수행하였다.  
<br>
### 2.1 XYZ 오차 분석
Eye-to-Hand 환경에서 ICP 기반 후정합을 적용하여 외재 파라미터의 정밀도를 개선하였다. 

그림: 오차 분포 시각화 (진한 색: 오차 큼 / 밝은 색: 오차 작음)

|  정확도 식별 실험 환경 | 초기 R,t 결과 | ICP 정합 적용 후 |
|-------------|----------------|--------------------|
| <img src="./data/asset/Environment1.png" width="300"/> | <img src="./data/asset/Eye-in-Hand2.png" width="300"/> | <img src="./data/asset/Eye-in-Hand_ICP_optimized.png" width="300"/> |

총 12개 지점에 대해 실험을 수행한 결과, 다음과 같은 **XYZ 오차 개선**을 확인:
| 구분   | X축 평균 오차 (mm) | Y축 평균 오차 (mm) | 전체 평균 오차 (mm) |
| ---- | ------------- | ------------- | ------------- |
| 정합 전 | 16.97         | 75.95         | **92.92**     |
| 정합 후 | 16.03         | 15.26         | **31.28**     |

* 전체 평균 오차 **66.34% 감소**
* 특히 Y축 오차는 **약 76.5%** 개선됨

> **정합 전 전체 오차**: 92.92 mm \
> **정합 후 전체 오차**: 31.28 mm \
> **오차 개선률**: **66.34%** \
> **Y축 오차 개선률이 두드러짐 (76.46%)**

<br>

### 2.2 RPY 오차 분석
추가적으로, RPY 각도의 정밀도를 평가하기 위해 **Table Normal Fix** 방식을 적용하였다.  
이 방식은 Depthmap 평면 분할로 얻은 테이블 법선 벡터를 이용해 회전 행렬을 보정하는 방법이다.  

그림: RPY 오차 개선 (좌: 보정 전, 우: 보정 후)

| 구분 | 보정 전 | 보정 후 |
|-----|---------|---------|
| 이미지 | <img src="./data/asset/table_normal_fix_before.png" width="300"/> | <img src="./data/asset/table_normal_fix_after.png" width="300"/> |
| camera RPY |  (1.460, 2.320, 93.034) [deg] | (1.460, 2.320, 93.034) [deg]  |
| Base RPY | (-168.07, -1.675, -2.52) [deg] | (-178.820, -0.69, -3.03) [deg] |

**축별 오차 및 개선률**

| 항목  | 보정 전 오차 (deg) | 보정 후 오차 (deg) | 개선률 |
|------|---------------------|---------------------|--------|
| Roll | 11.93               | 1.18                | **90.11%↓** |
| Pitch| 1.675               | 0.69                | **58.81%↓** |

**분석**  
- Table Normal Fix 적용 후, **Roll 값이 -180°에 가까워지고, Pitch 값이 0°에 더 근접**하였다.  
- 이는 Cube 대상 정렬 기준에서 **Roll이 ±180°에 가까울수록 오차가 작고, Pitch가 0°에 가까울수록 정확도가 높음**을 의미한다.  
- 결과적으로 RPY 보정에서 **Roll과 Pitch 오차가 개선**되어, 로봇 Pick & Place 수행 시 안정성이 향상됨을 확인하였다.  

<br>

## 3. YOLO 객체 탐지: x,y,z 좌표 추정

| 입력 이미지 | YOLO 탐지 결과 | ROI Depth 결과 |
|-------------|----------------|-----------|
| <img src="./data/asset/2025-08-04_15-27-57.jpg" width="300"/> | <img src="./data/asset/yolo_detect.png" width="300"/> |<img src="./data/asset/ROI_depth.png" width="300"/> |

- 이미지 상 bbox 중심점: **(x, y) = (196, 191)**
- 카메라 좌표계 기준 위치: **(X<sub>cam</sub>, Y<sub>cam</sub>, Z<sub>cam</sub>) = (-0.196, -0.046, 0.355) [m]**
- 로봇 Base 좌표계 변환 결과: **(X<sub>base</sub>, Y<sub>base</sub>, Z<sub>base</sub>) = (189.92, 136.7, -212.6) [mm]**
- ROI 영역은 YOLO Detection 결과를 바탕으로 추출된 bbox로부터 생성됨

<br>

## 4. RPY 자세 추정 (Surface Normal + PCA)
| 단계 | ROI PCD | Surface Normal 시각화 | Yaw 예측 결과 |
| --------------------- | --------------------- | --------------------- | --------------------- |
| **Before** | <img src="./data/asset/ROI_PCD.png" width="250"/>| <img src="./data/asset/surface_normal.png" width="250"/> |<img src="./data/asset/predict_yaw.png" width="250"/> |
| **After \(RANSAC 후)**  | <img src="./data/asset/ROI_PCD_after_RANSAC.png" width="250"/> | <img src="./data/asset/surface_normal_after_RNASAC.png" width="250"/> | <img src="./data/asset/predict_yaw_after_RANSAC.png" width="250"/> |

- **Roll, Pitch**는 평균 Surface Normal을 기반으로 추정
- **Yaw**는 PCA의 단축(minor axis) 방향을 기반으로 추정
- **RANSAC 기반 평면 분리** 적용 전후 비교:
  - Yaw 오차 개선: 36.25° → 8.06° (77.8% 감소) 

