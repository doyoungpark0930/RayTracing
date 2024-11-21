# [RayTracing을 이용한 큐브매핑 및 간단한 PhotonMapping]<br>
<small>개발환경 : WindowAPI, visual Studio, GLM함수 사용</small><br>

<img width="958" alt="photonMapping" src="https://github.com/user-attachments/assets/d0bc9637-de89-4691-9b05-2cace86d0780">

## 큐브맵 이미지
![canyon1](https://github.com/user-attachments/assets/be2ce931-dc6f-4cb4-ab1a-8340eb3f4e82)
<small>큐브맵 이미지 출처 : https://paulbourke.net/panorama/cubemaps/</small>

- 투영 방식 : perspective projection

- 왼쪽 구 : 100% 투명 유리구슬
- 중앙 구 : 100% 반사 유리구슬
- 오른쪽 구 : 100% 투명 유리 구슬

사각형의 texturing은 삼각형 두개를 형성하고 각 uv좌표를 barycentric coordinate를 이용 함.<br><br>

Ray를 재귀 형식으로 쏘아서, 반사와 굴절 나타냄<br>
굴절은 스넬의 법칙, 유리와 공기의 굴절 비율 은 1.5를 이용<br><br>


caustic ball을 표현하는데 PhotonMapping을 이용하였으나, 간단하게만 구현 하였음.<br>

[PhotonMapping Idea]
1. 바닥에 시점 ray를 쏘았을 때, 그 맞은 지점에서 각 유리구슬로 일정 빛의 세기를 갖고 있는 광자를 여러개 쏜다
2. 광자는 반드시 구 안 쪽을 지나가도록 설정한다(어차피 광자가 구 바깥으로 나가면 사라지는 것과 차이가 없기 때문)
3. Light는 매우 먼 거리에 설정한다.(Directional Light의 효과를 내기 위함)
4. 광자가 유리 구슬에 두번 굴절되고 구슬 바깥으로 나가면, 그 ray의 방향과 ray의 시작점과 light를 바라본 벡터의 내적하여 각도가 작을 수록 빛이 줄지 않도록 한다
5. 그 내적을 pow()를 이용해 하이라이트 효과를 낸다
