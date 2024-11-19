#pragma once

#include "Sphere.h"
#include "Ray.h"
#include "Light.h"
#include "Triangle.h"
#include "Square.h"
#include <cmath>
#include <vector>
#include <glm/gtc/random.hpp>


namespace hlab
{

	using namespace std;
	using namespace glm;

	class Raytracer
	{
	public:
		int width, height;
		Light light;
		vector<shared_ptr<Object>> objects;

		Raytracer(const int& width, const int& height)
			: width(width), height(height)
		{
			auto sphere1 = make_shared<Sphere>(vec3(-3.5f, -1.2f, 5.0f), 0.8f);

			sphere1->amb = vec3(0.2f);
			sphere1->dif = vec3(0.0f, 0.0f, 0.0f);
			sphere1->spec = vec3(1.0f);
			sphere1->alpha = 20.0f;
			sphere1->reflection = 0.0f;
			sphere1->transparency = 1.0f;
			sphere1->type = ObjectType::Sphere1;

			objects.push_back(sphere1);

			auto sphere2 = make_shared<Sphere>(vec3(2.0f, -1.2f, 2.5f), 0.8f);

			sphere2->amb = vec3(0.2f);
			sphere2->dif = vec3(0.0f, 0.0f, 0.0f);
			sphere2->spec = vec3(1.0f);
			sphere2->alpha = 20.0f;
			sphere2->reflection = 0.0f;
			sphere2->transparency = 1.0f;
			sphere2->type = ObjectType::Sphere2;

			objects.push_back(sphere2);
			
			
			auto sphere3 = make_shared<Sphere>(vec3(0.0f, -0.8f, 3.5f), 1.2f);

			sphere3->amb = vec3(0.2f);
			sphere3->dif = vec3(0.0f, 0.0f, 0.0f);
			sphere3->spec = vec3(1.0f);
			sphere3->alpha = 20.0f;
			sphere3->reflection = 1.0f;
			sphere3->transparency = 0.0f;

			objects.push_back(sphere3);
			
			
			
			std::vector<vec3> textureImage(1 * 1, vec3(0.2f, 0.2f, 0.2f));
			auto groundTexture = std::make_shared<Texture>(1, 1, textureImage);

			auto ground = make_shared<Square>(vec3(-6.0f, -2.0f, 0.0f), vec3(-6.0f, -2.0f, 7.0f), vec3(6.0f, -2.0f, 7.0f), vec3(6.0f, -2.0f, 0.0f),
				vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 1.0f));

			ground->amb = vec3(1.0f);
			ground->dif = vec3(0.0f);
			ground->spec = vec3(0.5f);
			ground->alpha = 30.0f;
			ground->reflection = 0.3f;
			ground->ambTexture = groundTexture;
			ground->difTexture = groundTexture;
			ground->type = ObjectType::Ground;

			objects.push_back(ground);
			
			
			

			float cubeLen = 10.0f;

			auto frontTexture = std::make_shared<Texture>("front.jpg");
			auto front = make_shared<Square>(vec3(-cubeLen, cubeLen, cubeLen), vec3(cubeLen, cubeLen, cubeLen), vec3(cubeLen, -cubeLen, cubeLen), vec3(-cubeLen, -cubeLen, cubeLen),
				vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 1.0f));

			front->amb = vec3(1.0f);
			front->dif = vec3(0.0f);
			front->spec = vec3(0.0f);
			front->alpha = 10.0f;
			front->reflection = 0.0f;
			front->ambTexture = frontTexture;
			front->difTexture = frontTexture;

			objects.push_back(front);

			auto rightTexture = std::make_shared<Texture>("right.jpg");
			auto right = make_shared<Square>(vec3(cubeLen, cubeLen, cubeLen), vec3(cubeLen, cubeLen, -cubeLen), vec3(cubeLen, -cubeLen, -cubeLen), vec3(cubeLen, -cubeLen, cubeLen),
				vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 1.0f));

			right->amb = vec3(1.0f);
			right->dif = vec3(0.0f);
			right->spec = vec3(0.0f);
			right->alpha = 10.0f;
			right->reflection = 0.0f;
			right->ambTexture = rightTexture;
			right->difTexture = rightTexture;

			objects.push_back(right);


			auto leftTexture = std::make_shared<Texture>("left.jpg");
			auto left = make_shared<Square>(vec3(-cubeLen, cubeLen, -cubeLen), vec3(-cubeLen, cubeLen, cubeLen), vec3(-cubeLen, -cubeLen, cubeLen), vec3(-cubeLen, -cubeLen, -cubeLen),
				vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 1.0f));

			left->amb = vec3(1.0f);
			left->dif = vec3(0.0f);
			left->spec = vec3(0.0f);
			left->alpha = 10.0f;
			left->reflection = 0.0f;
			left->ambTexture = leftTexture;
			left->difTexture = leftTexture;

			objects.push_back(left);

			auto topTexture = std::make_shared<Texture>("top.jpg");
			auto top = make_shared<Square>(vec3(-cubeLen, cubeLen, -cubeLen), vec3(cubeLen, cubeLen, -cubeLen), vec3(cubeLen, cubeLen, cubeLen), vec3(-cubeLen, cubeLen, cubeLen),
				vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 1.0f));

			top->amb = vec3(1.0f);
			top->dif = vec3(0.0f);
			top->spec = vec3(0.0f);
			top->alpha = 10.0f;
			top->reflection = 0.0f;
			top->ambTexture = topTexture;
			top->difTexture = topTexture;

			objects.push_back(top);


			auto bottomTexture = std::make_shared<Texture>("bottom.jpg");
			auto bottom = make_shared<Square>(vec3(-cubeLen, -cubeLen, cubeLen), vec3(cubeLen, -cubeLen, cubeLen), vec3(cubeLen, -cubeLen, -cubeLen), vec3(-cubeLen, -cubeLen, -cubeLen),
				vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 1.0f));

			bottom->amb = vec3(1.0f);
			bottom->dif = vec3(0.0f);
			bottom->spec = vec3(0.0f);
			bottom->alpha = 10.0f;
			bottom->reflection = 0.0f;
			bottom->ambTexture = bottomTexture;
			bottom->difTexture = bottomTexture;


			objects.push_back(bottom);

			auto backTexture = std::make_shared<Texture>("back.jpg");
			auto back = make_shared<Square>(vec3(cubeLen, cubeLen,- cubeLen), vec3(-cubeLen, cubeLen, -cubeLen), vec3(-cubeLen, -cubeLen, -cubeLen), vec3(cubeLen, -cubeLen, -cubeLen),
				vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 1.0f));

			back->amb = vec3(1.0f);
			back->dif = vec3(0.0f);
			back->spec = vec3(0.0f);
			back->alpha = 10.0f;
			back->reflection = 0.0f;
			back->ambTexture = backTexture;
			back->difTexture = backTexture;

			objects.push_back(back);




			light = Light{ {2.0f, 800.0f, 500.0f} }; // 화면 앞쪽

		}

		Hit FindClosestCollision(Ray& ray)
		{
			float closestD = 1000.0; // inf
			Hit closestHit = Hit{ -1.0, dvec3(0.0), dvec3(0.0) };

			for (int l = 0; l < objects.size(); l++)
			{
				auto hit = objects[l]->CheckRayCollision(ray);

				if (hit.d >= 0.0f)
				{
					if (hit.d < closestD)
					{
						closestD = hit.d;
						closestHit = hit;
						closestHit.obj = objects[l];

						// 텍스춰 좌표
						closestHit.uv = hit.uv;
					}
				}
			}

			return closestHit;
		}

		float Saturate(float x) { return glm::max(0.0f, glm::min(1.0f, x)); }

		float CalcAttenuation(float d, float falloffStart, float falloffEnd) {
			// Linear falloff
			return Saturate((falloffEnd - d) / (falloffEnd - falloffStart));
		}

		float causticsRay(Ray& ray, float lightStrength, const int recurseLevel) //무조건 처음 hit.d>0임
		{
			if (recurseLevel <= 0)
			{
				const vec3 dirToLight = glm::normalize(light.pos - ray.start);
				lightStrength *= glm::pow(glm::max(dot(dirToLight, ray.dir), 0.0f), 50.0f);

				//여기서 빛과 ray의각도로 float반환
				return lightStrength;
			}
				

			const auto hit = FindClosestCollision(ray);

			if (hit.d >= 0.0f)
			{
				if (hit.obj->transparency)
				{
					const float ior = 1.5f; // Index of refraction (유리: 1.5, 물: 1.3)

					float eta; // sinTheta1 / sinTheta2
					vec3 normal;

					if (glm::dot(ray.dir, hit.normal) < 0.0f) // 밖에서 안으로 들어가는 경우 (예: 공기->유리)
					{
						eta = ior;
						normal = hit.normal;
					}
					else // 안에서 밖으로 나가는 경우 (예: 유리->공기)
					{
						eta = 1.0f / ior;
						normal = -hit.normal;
					}

					const float cosTheta1 = -dot(normal, ray.dir);
					const float sinTheta1 = sqrt(1 - cosTheta1 * cosTheta1);
					const float sinTheta2 = sinTheta1 / eta;
					const float cosTheta2 = sqrt(1 - sinTheta2 * sinTheta2);

					const vec3 m = glm::normalize(dot(normal, -ray.dir) * normal + ray.dir);
					const vec3 a = -normal * cosTheta2;
					const vec3 b = m * sinTheta2;
					const vec3 refractedDirection = glm::normalize(a + b); // transmission

					Ray nextRay = { hit.point + refractedDirection * 1e-4f, refractedDirection };
					float fallOffStart = 0.0f;
					float fallOffEnd = 1.7f; //구의 지름이상이어야함
					float att = CalcAttenuation(hit.d, fallOffStart, fallOffEnd);

					lightStrength *= att;
					lightStrength *= hit.obj->transparency;
					lightStrength *= causticsRay(nextRay, lightStrength, recurseLevel - 1);


				}
				else
				{
					lightStrength = 0.0f;
				}

			}

			return lightStrength;


		}

		// 광선이 물체에 닿으면 그 물체의 색 반환
		vec3 traceRay(Ray& ray, const int recurseLevel)
		{
			if (recurseLevel < 0)
				return vec3(0.0f);

			// Render first hit
			const auto hit = FindClosestCollision(ray);

			if (hit.d >= 0.0f)
			{
				glm::vec3 color(0.0f);


				// Diffuse
				const vec3 dirToLight = glm::normalize(light.pos - hit.point);

				glm::vec3 phongColor(0.0f);

				const float diff = glm::max(dot(hit.normal, dirToLight), 0.0f);

				// Specular
				const vec3 reflectDir = hit.normal * 2.0f * dot(dirToLight, hit.normal) - dirToLight;
				const float specular = glm::pow(glm::max(glm::dot(-ray.dir, reflectDir), 0.0f), hit.obj->alpha);

				if (hit.obj->ambTexture)
				{
					phongColor += hit.obj->amb * hit.obj->ambTexture->SampleLinear(hit.uv);
				}
				else
				{
					phongColor += hit.obj->amb;
				}

				if (hit.obj->difTexture)
				{
					phongColor += diff * hit.obj->dif * hit.obj->difTexture->SampleLinear(hit.uv);
				}
				else
				{
					phongColor += diff * hit.obj->dif;
				}

				phongColor += hit.obj->spec * specular; 

				if (hit.obj->type == ObjectType::Ground)
				{
			

					const vec3 sphere1Center = vec3(-3.5f, -1.2f, 5.0f); // 예시 값, 실제로는 유리 구슬의 중심 좌표
					const float sphere1Radius = 0.8f; // 예시 값, 실제로는 구의 반경

					const vec3 sphere2Center = vec3(2.0f, -1.2f, 2.5f); // 예시 값, 실제로는 유리 구슬의 중심 좌표
					const float sphere2Radius = 0.8f; // 예시 값, 실제로는 구의 반경
			
					// Hit 지점에서 구슬 중심을 향하는 벡터를 계산
					vec3 directionToSphere1 = sphere1Center - hit.point;
					vec3 directionToSphere2 = sphere2Center - hit.point;
					
					const int numRays = 40;
					for (int i = 0; i < numRays; i++)
					{
						float lightStrength = 1.3f;//광자 하나의 세기

						float theta = glm::linearRand(0.0f, glm::two_pi<float>());
						float phi = glm::acos(glm::linearRand(-1.0f, 1.0f));

						// 구면 좌표를 직교 좌표로 변환
						const vec3 randomDir1 = vec3(
							std::sin(phi) * std::cos(theta),
							std::sin(phi) * std::sin(theta),
							std::cos(phi)
						) * (sphere1Radius - 1e-4f); //구보다 조금 안 쪽으로 향하게해야 반드시 2번 맞음

						const vec3 randomDir2 = vec3(
							std::sin(phi) * std::cos(theta),
							std::sin(phi) * std::sin(theta),
							std::cos(phi)
						) * (sphere2Radius - 1e-4f); //구보다 조금 안 쪽으로 향하게해야 반드시 2번 맞음

						// 방향 벡터를 hit 지점에서 구의 중심으로 향하는 벡터에 기반해 변형
						const vec3 rayDirection1 = glm::normalize(directionToSphere1 + randomDir1);
						const vec3 rayDirection2 = glm::normalize(directionToSphere2 + randomDir2);

						Ray photonRay1 = { hit.point + rayDirection1 * 1e-4f, rayDirection1 };
						Ray photonRay2 = { hit.point + rayDirection2 * 1e-4f, rayDirection2 };
						
						//ray가 해당 구에 안 맞고 다른 곳에 맞으면 무효화 시키기 위함
						Hit hit1 = FindClosestCollision(photonRay1);
						Hit hit2 = FindClosestCollision(photonRay2);
						if (hit1.obj->type == ObjectType::Sphere1) {
							phongColor += causticsRay(photonRay1, lightStrength, 2) * vec3(1.0f, 1.0f, 1.0f);
						}
						if (hit2.obj->type == ObjectType::Sphere2)
						{
							phongColor += causticsRay(photonRay2, lightStrength, 2) * vec3(1.0f, 1.0f, 1.0f);
						}

						phongColor = glm::clamp(phongColor, vec3(0.0f), vec3(1.0f));

					}

					


						
					
				}

				color += phongColor * (1.0f - hit.obj->reflection - hit.obj->transparency);

				if (hit.obj->reflection)
				{
					const auto reflectedDirection = glm::normalize(2.0f * hit.normal * dot(-ray.dir, hit.normal) + ray.dir);
					Ray reflection_ray{ hit.point + reflectedDirection * 1e-4f, reflectedDirection }; // add a small vector to avoid numerical issue

					color += traceRay(reflection_ray, recurseLevel - 1) * hit.obj->reflection;
				}

				// 참고
				// https://samdriver.xyz/article/refraction-sphere (그림들이 좋아요)
				// https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel (오류있음)
				// https://web.cse.ohio-state.edu/~shen.94/681/Site/Slides_files/reflection_refraction.pdf (슬라이드가 보기 좋지는 않지만 정확해요)
				if (hit.obj->transparency)
				{
					const float ior = 1.5f; // Index of refraction (유리: 1.5, 물: 1.3)

					float eta; // sinTheta1 / sinTheta2
					vec3 normal;

					if (glm::dot(ray.dir, hit.normal) < 0.0f) // 밖에서 안으로 들어가는 경우 (예: 공기->유리)
					{
						eta = ior;
						normal = hit.normal;
					}
					else // 안에서 밖으로 나가는 경우 (예: 유리->공기)
					{
						eta = 1.0f / ior;
						normal = -hit.normal;
					}

					const float cosTheta1 = -dot(normal, ray.dir);
					const float sinTheta1 = sqrt(1 - cosTheta1 * cosTheta1);
					const float sinTheta2 = sinTheta1 / eta;
					const float cosTheta2 = sqrt(1 - sinTheta2 * sinTheta2);

					const vec3 m = glm::normalize(dot(normal, -ray.dir) * normal + ray.dir);
					const vec3 a = -normal * cosTheta2;
					const vec3 b = m * sinTheta2;
					const vec3 refractedDirection = glm::normalize(a + b); // transmission

					Ray nextRay = { hit.point + refractedDirection * 1e-4f, refractedDirection };
					color += traceRay(nextRay, recurseLevel - 1) * hit.obj->transparency;

					// Fresnel 효과는 생략되었습니다.
				}

				return color;
			}

			return vec3(0.0f);
		}

		void Render(std::vector<glm::vec4>& pixels)
		{
			std::fill(pixels.begin(), pixels.end(), vec4(0.0f, 0.0f, 0.0f, 1.0f));

			const vec3 eyePos(0.0f, 0.0f, -1.5f); //원래 0.0,0.0,-1.5

#pragma omp parallel for
			for (int j = 0; j < height; j++)
				for (int i = 0; i < width; i++)
				{
					const vec3 pixelPosWorld = TransformScreenToWorld(vec2(i, j));
					Ray pixelRay{ pixelPosWorld, glm::normalize(pixelPosWorld - eyePos) };
					pixels[i + width * j] = vec4(glm::clamp(traceRay(pixelRay, 5), 0.0f, 1.0f), 1.0f);
				}
		}

		vec3 TransformScreenToWorld(vec2 posScreen)
		{
			const float xScale = 2.0f / this->width;
			const float yScale = 2.0f / this->height;
			const float aspect = float(this->width) / this->height;

			// 3차원 공간으로 확장 (z좌표는 0.0)
			return vec3((posScreen.x * xScale - 1.0f) * aspect, -posScreen.y * yScale + 1.0f, 0.0f);
		}
	};
}