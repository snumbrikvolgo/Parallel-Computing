#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>
#include "geometry.h"
#include "mpi.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


const float EPSILON = 0.0000001;
const float FLARE_COEFF = 0.5, FLARE_POW = 100;
const float LIGHT_BASE = 100, LIGHT_COEFF = 0.3;
const int DEPTH = 5;

void saveImage(char* filename, int w, int h, unsigned char* data) {
	const int channels_num = 1;
	stbi_write_jpg(filename, w, h, channels_num, data, 100);
}

struct Light {
    Light(const Vec3f &p, const float &i) : position(p), intensity(i) {}
    Vec3f position;
    float intensity;
};

Vec3f reflect(const Vec3f &I, const Vec3f &N) {
    return I - N*2.f*(I*N);
}


// struct Material {
//     Material(const Vec2f &a, const Vec3f &color, const float &spec) : albedo(a), diffuse_color(color), specular_exponent(spec) {}
//     Material() : albedo(1,0), diffuse_color(), specular_exponent() {}
//     Vec2f albedo;
//     Vec3f diffuse_color;
//     float specular_exponent;
// };

struct Triangle{
    const float EPSILON = 1e-8;
    Vec3f vert0;
    Vec3f vert1;
    Vec3f vert2;
    Vec3f normal;
    //Material material;

    Triangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2) : vert0(v0), vert1(v1), vert2(v2) {
      normal = cross(v1 - v0, v2 - v0).normalize();
    }
    //Moller - Trumbor algorithm
    bool ray_intersect(const Vec3f &orig, const Vec3f &dir, float &t, int& transparent) const {
    float u; float v; float w;
    Vec3f edge1 = vert1 - vert0;
    Vec3f edge2 = vert2 - vert0;

    Vec3f pvec = cross(dir, edge2);

    float det = edge1 * pvec;

    if (det < EPSILON)
        return false;
    float inv_det = 1.0f / det;

    Vec3f tvec = orig - vert0;

    u = (tvec * pvec) * inv_det;

    if (u < 0.0 || u > 1.0f)
        return false;

    Vec3f qvec = cross(tvec, edge1);

    v = (dir * qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0f)
        return false;

    t = (edge2 * qvec) * inv_det;
    if (t > 0.0)
    {
      float s = (dir - orig) * normal;
      transparent = s < 0.0 ? -1 : 1;
    }
    return true;
  }
};
void read_triangles(std::string file_name, std::vector<Triangle>& triangles, int size)
{
  std::ifstream inf(file_name);
  //Material red_rubber(Vec2f(0.9,  0.1), Vec3f(0.3, 0.1, 0.1),   50.);
  Vec3f a,b,c;

  while (inf && size--)
  {
    inf >> a.x >> a.y >> a.z;
    inf >> b.x >> b.y >> b.z;
    inf >> c.x >> c.y >> c.z;
    triangles.push_back(Triangle(a,b,c));
  }

  return;
}


float cast_ray(const Vec3f &orig, const Vec3f &dir, const std::vector<Triangle> &triangles,
               const std::vector<Light> &lights, int recursion)
{
  if (recursion > DEPTH)
    return 0.0;
  recursion++;

  float ret = 0;
  float dist = 0.0;
  int transparent = 0;


  for (size_t i = 0; i < triangles.size(); i++)
  {
    bool intersects = triangles[i].ray_intersect(orig, dir, dist, transparent);
    Vec3f point = orig + dir * dist;
    if (intersects)
    {
      if (transparent == 1)
      {
        float flare = powf(fabs(triangles[i].normal * (point - lights[0].position).normalize()), FLARE_POW);
				ret += FLARE_COEFF * flare + LIGHT_COEFF * fabs(triangles[i].normal * (point - lights[0].position).normalize()) + \
							cast_ray(point + dir * 0.001, dir, triangles, lights, recursion);
      }
      else
      {
        float flare = powf(fabs(triangles[i].normal * (point - lights[0].position).normalize()), FLARE_POW);
				ret += FLARE_COEFF * flare + LIGHT_COEFF * fabs(triangles[i].normal * (point - lights[0].position).normalize()) + \
							cast_ray(point - dir * 0.001, reflect(dir, triangles[i].normal), triangles, lights, recursion);
      }
    }
  }
  return ret;
}



void render(const std::vector<Triangle> &triangles, const std::vector<Light> &lights, int argc, char* argv[]) {
    const int width    = 1400;
    const int height   = 1000;
    const int fov      = 3.14159 / 1.7;

		int size, rank;

		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		double time;

    unsigned char data[height * width];
    memset(data, 0,height * width);

    Vec3f dir, camera(10, 1, -6), rotation(-2, 0, 1);
    float brightness = 0.0;
    //std::cout << triangles.size() << "\n";

		if (rank == 0) {
			std::cout << "Cluster size: " << size << std::endl;
			std::cout << "Triangles amount: " << triangles.size() << std::endl;
			time = MPI_Wtime();
		}

    for (size_t j = height / size * rank; j < height / size * (rank + 1); j++)
    {
      for (size_t i = 0; i < width; i++)
      {
        float x = (2 * (i + 0.5) / (float)width - 1) * tan(fov / 2.) * width / (float)height;
        float y = -(2 * (j + 0.5) / (float)height - 1) * tan(fov / 2.);
        dir = (Vec3f(x, y, 1) + rotation).normalize();

        brightness = LIGHT_BASE * (cast_ray(camera, dir, triangles, lights, 0));
        brightness = brightness > 255 ? (char)255 : (char)brightness;
        data[i + j * width] = brightness;
      }
    }
		MPI_Gather(&data[height / size * rank * width], height * width / size, MPI_CHAR, data, height * width / size, MPI_CHAR, 0, MPI_COMM_WORLD);
    std::cout << "kek\n";

		if (rank == 0) {
				std::cout << "Time: " << MPI_Wtime() - time << std::endl;
				char pic[] = "result.jpg";
				saveImage(pic, width, height, data);
			}
		MPI_Finalize();
}

int main(int argc, char* argv[]) {
    std::vector<Triangle> triangles;
    read_triangles("model.xyz", triangles, 100);
    std::vector<Light>  lights;
    lights.push_back(Light(Vec3f(5, -5, 0), 10));
    render(triangles, lights, argc, argv);

    return 0;
}
