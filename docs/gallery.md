# AutoDex Homepage and Dataset Gallery

## 목표
AutoDex 논문 landing page와 dataset gallery를 만든다. 전체 구조는 HRDexDB류의 dataset site를 참고하되, object list가 먼저 압도적으로 보이는 방식보다는 선택된 episode를 보여주는 main panel이 중심이 되도록 설계한다.

## 페이지 구성

### 1. Landing Page
논문, 데이터셋, task 정의, 대표 결과를 빠르게 전달하는 진입 페이지로 만든다.

핵심 콘텐츠:
- AutoDex 개요
- grasp/task가 무엇인지 설명
- scene과 capture setup 설명
- dataset 규모 요약: objects, poses/grasps, episodes, views, hands
- 대표 video 또는 interactive 3D preview
- paper / code / dataset / gallery link

처음부터 object thumbnail grid를 길게 보여주는 구성은 우선순위가 낮다. 메인 메시지와 대표 panel이 먼저 보여야 한다.

### 2. Gallery Page
선택된 object, pose, episode, view를 main panel에서 크게 보여주는 페이지로 만든다. filter와 selector는 panel 위 또는 옆에 두되, 시각적으로 가볍게 유지한다.

기본 레이아웃:
```text
[compact controls: object / pose / hand / success / view / mode]
[large main panel: video or interactive 3D]
[episode metadata + trajectory/success summary]
[optional thumbnail drawer]
```

## Interaction 설계

### Main Panel
기본 상태에서는 선택된 episode의 촬영 영상을 보여준다. 사용자가 mode를 바꾸면 interactive 3D reconstruction 또는 3D panel을 보여준다.

지원할 모드:
- captured video
- synchronized multi-view video
- selected single-view video
- video + trajectory overlay
- interactive 3D reconstruction
- object mesh / pose preview

재생 관련 기능:
- playback bar
- play/pause
- frame 또는 timestamp 표시
- view switching
- overlay on/off

### Object/Pose Selector
object와 pose thumbnail은 항상 펼쳐두지 않고, 선택할 때만 보이도록 한다. UI는 toggle/drawer/dropdown 형태가 적합하다.

object selector:
- object name/id 검색
- category/material 등 metadata 기반 filter
- 선택 시 object thumbnail 또는 3D asset preview 표시

pose selector:
- pose id 또는 grasp id 선택
- 선택 시 object와 접촉한 3D pose를 turntable preview로 표시
- success/fail episode 수 요약

### Filters
초기 구현에서 필요한 filter:
- object
- pose/grasp
- hand type
- success only
- fail only
- all episodes
- view id
- video / 3D mode

추가 확장 후보:
- material/category
- difficulty
- capture version
- validation status
- reconstruction available
- trajectory available

## 필요한 데이터 연결
Gallery 구현에는 다음 asset이 필요하다.

- 20개 또는 대표 multi-view video set
- synchronized robot trajectory
- success label
- object tracking 기반 trajectory
- selected view별 video URL
- overlay preview asset
- 3D reconstruction asset
- object mesh GLB
- pose turntable preview
- object/pose/episode metadata manifest

robot trajectory는 object tracking을 통해 산출해야 한다. trajectory가 없는 episode는 gallery에서 비활성화하거나 “trajectory unavailable” 상태로 구분한다.

## 구현 순서
1. metadata schema 확정
2. landing page 정보 구조 작성
3. gallery static prototype 제작
4. main panel video playback 구현
5. compact selector/filter 구현
6. object/pose thumbnail drawer 구현
7. success/fail filter 연결
8. synchronized trajectory와 overlay 표시
9. interactive 3D reconstruction mode 연결
10. 배포용 asset 경로와 GitHub Pages 배포 흐름 정리

## 산출물
- landing page draft
- gallery page prototype
- gallery metadata schema
- video/reconstruction/mesh asset path convention
- selector/filter UI 구현
- main panel playback + overlay + 3D mode
- GitHub Pages 배포 가능한 정적 사이트
