#include "rdr/integrator.h"

#include <omp.h>

#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        // TODO(HW3): generate #spp rays for each pixel and use Monte Carlo
        // integration to compute radiance.
        //
        // Useful Functions:
        //
        // @see Sampler::getPixelSample for getting the current pixel sample
        // as Vec2f.
        //
        // @see Camera::generateDifferentialRay for generating rays given
        // pixel sample positions as 2 floats.

        // You should assign the following two variables
        // const Vec2f &pixel_sample = ...
        // auto ray = ...
        const Vec2f &pixel_sample = sampler.getPixelSample();
        auto ray =
            camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);

        // After you assign pixel_sample and ray, you can uncomment the
        // following lines to accumulate the radiance to the film.
        //
        //
        // Accumulate radiance
        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f accumulated_color(0.0f);
  Vec3f throughput(
      1.0f);  // Represents how much light is transmitted along the path

  for (int i = 0; i < max_depth; ++i) {
    SurfaceInteraction interaction;
    bool intersected = scene->intersect(ray, interaction);

    if (!intersected) {
      // Ray escaped the scene, contribute nothing.
      break;
    }

    // Set the outgoing direction for BSDF evaluation
    interaction.wo = -ray.direction;

    // Perform RTTI to determine the type of the surface
    const BSDF *bsdf = interaction.bsdf;
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(bsdf) != nullptr;

    if (is_ideal_diffuse) {
      // Hit a diffuse surface. Calculate direct lighting here and terminate the
      // path.
      Vec3f direct_light =
          directLighting(scene, interaction);  // Contribution from point light

      // Add contribution from area lights using multiple samples per light
      const int light_samples =
          16;  // Number of samples per area light, can be adjusted
      for (const auto &light : scene->getLights()) {
        Vec3f light_contribution(0.0f);
        for (int j = 0; j < light_samples; ++j) {
          SurfaceInteraction light_sample = light->sample(sampler);
          Vec3f light_dir                 = light_sample.p - interaction.p;
          Float dist2                     = Dot(light_dir, light_dir);
          light_dir                       = Normalize(light_dir);

          DifferentialRay shadow_ray(interaction.p, light_dir);
          shadow_ray.setTimeMax(std::sqrt(dist2) - 1e-4f);
          SurfaceInteraction shadow_isect;
          if (scene->intersect(shadow_ray, shadow_isect)) {
            continue;
          }

          interaction.wi  = light_dir;
          Vec3f f         = bsdf->evaluate(interaction);
          Vec3f Le        = light->Le(light_sample, -light_dir);
          Float cos_theta = std::max(0.0f, Dot(interaction.normal, light_dir));
          Float cos_light =
              std::max(0.0f, Dot(light_sample.normal, -light_dir));
          Float pdf = light->pdf(light_sample);
          if (pdf > 0.0f) {
            light_contribution +=
                f * Le * cos_theta * cos_light / (dist2 * pdf);
          }
        }
        direct_light +=
            light_contribution / light_samples;  // Average the contribution
      }

      accumulated_color += throughput * direct_light;
      break;  // Path terminates on diffuse surfaces in this simple integrator.
    } else if (is_perfect_refraction) {
      // Hit a refractive surface. Sample a new direction and continue the path.
      Float pdf;
      Vec3f f = bsdf->sample(
          interaction, sampler, &pdf);  // This updates interaction.wi

      // Update throughput by the BSDF value and PDF
      if (pdf > 0.0f) {
        throughput *= f / pdf;
      } else {
        break;  // Stop if sampling is invalid
      }

      // Create the next ray and continue the loop
      ray = interaction.spawnRay(interaction.wi);
    } else {
      // Hit a surface we don't handle (or a light source directly). Stop.
      break;
    }
  }

  return accumulated_color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  Vec3f color(0, 0, 0);
  Float dist_to_light = Norm(point_light_position - interaction.p);
  Vec3f light_dir     = Normalize(point_light_position - interaction.p);
  auto test_ray       = DifferentialRay(interaction.p, light_dir);

  // TODO(HW3): Test for occlusion
  //
  // You should test if there is any intersection between interaction.p and
  // point_light_position using scene->intersect. If so, return an occluded
  // color. (or Vec3f color(0, 0, 0) to be specific)
  //
  // You may find the following variables useful:
  //
  // @see bool Scene::intersect(const Ray &ray, SurfaceInteraction &interaction)
  //    This function tests whether the ray intersects with any geometry in the
  //    scene. And if so, it returns true and fills the interaction with the
  //    intersection information.
  //
  //    You can use iteraction.p to get the intersection position.
  //
  SurfaceInteraction shadow_interaction;
  test_ray.setTimeMax(
      dist_to_light - 1e-4);  // Avoid self-intersection at light source
  if (scene->intersect(test_ray, shadow_interaction)) {
    return color;  // Occluded
  }
  // Not occluded, compute the contribution using perfect diffuse diffuse model
  // Perform a quick and dirty check to determine whether the BSDF is ideal
  // diffuse by RTTI
  const BSDF *bsdf      = interaction.bsdf;
  bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

  if (bsdf != nullptr && is_ideal_diffuse) {
    // TODO(HW3): Compute the contribution
    //
    // You can use bsdf->evaluate(interaction) * cos_theta to approximate the
    // albedo. In this homework, we do not need to consider a
    // radiometry-accurate model, so a simple phong-shading-like model is can be
    // used to determine the value of color.

    // The angle between light direction and surface normal
    Float cos_theta =
        std::max(Dot(light_dir, interaction.normal), 0.0f);  // one-sided

    // Define a default light intensity (can be adjusted as needed)

    // You should assign the value to color
    color = bsdf->evaluate(interaction) * cos_theta * point_light_flux /
            (dist_to_light * dist_to_light);
  }

  return color;
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li(  // NOLINT
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

RDR_NAMESPACE_END
