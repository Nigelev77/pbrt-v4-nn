#import "common.typ": *

#set page(margin: (left: 0.25cm, right: 0.25cm, top: 1cm, bottom: 1cm))


#align(center)[
  #set text(size: 16pt)
  Appendix
]

//let model-inputs-outputs =

//
/*
  - Ray origin + direction
  - Maximum $t$ value until ray intersection
  - $beta$ transmission value (explained later)

  The outputs, or targets for the model, are:
  - Intrinsic emitted Radiance ($L_"intrinsic"$) added from the medium
  - Weighted Throughput Ratio $T$ of the ray after passing through the medium


*/


#let model-inputs-outputs = figure(
  {
    set text(size: 9.5pt)
    table(
      columns: (auto, auto),
      align: horizon,
      table.header([*Inputs*], [*Outputs*]),
      [Ray Origin $x,y,z$], [Volumetric emitted radiance $L_"intrinsic"$],
      [Ray Direction $x,y,z$], [Weighted Throughput Ratio $T$],
      [`tMax` parametric value until ray intersection],
    )
  },
  caption: [Instant-NGP model Inputs and Outputs],
)

#model-inputs-outputs



#let gt-views = image-figure(
  ("../images/GT_view1.png", [View 1]),
  ("../images/GT_view2.png", [View 2]),
  ("../images/GT_view3.png", [View 3]),
  columns: 3,
  caption: [Inference for Native PBRT path on 3 determined novel views],
)

#gt-views

#let spectral-views = image-figure(
  ("../images/spectral_view1.png", [View 1]),
  ("../images/spectral_view2.png", [View 2]),
  ("../images/spectral_view3.png", [View 3]),
  columns: 3,
  caption: [Inference for Spectral on 3 determined novel views],
)

#spectral-views


#let neurons128-layers3-color = image-figure(
  ("../images/large_color_view1.png", [View 1]),
  ("../images/large_color_view2.png", [View 2]),
  ("../images/large_color_view3.png", [View 3]),
  columns: 3,
  caption: [Inference for 128 neurons 3 layers color on 3 determined novel views],
)

#neurons128-layers3-color

#let neurons128-layers3-T = image-figure(
  ("../images/test_larger_view1.png", [View 1]),
  ("../images/test_larger_view2.png", [View 2]),
  ("../images/test_larger_view3.png", [View 3]),
  columns: 3,
  caption: [Inference for 128 neurons 3 layers T only on 3 determined novel views],
)

#neurons128-layers3-T


#let nlevels12-hashgrid19 = image-figure(
  ("../images/medium_tAfter_linear_T_view1.png", [View 1]),
  ("../images/medium_tAfter_linear_T_view2.png", [View 2]),
  ("../images/medium_tAfter_linear_T_view3.png", [View 3]),
  columns: 3,
  caption: [Inference for 12 nlevels and log hashgrid size 19 on 3 determined novel views],
)

#nlevels12-hashgrid19

#let timings-graph = image-figure(
  "../images/timings_graph.png",
  columns: 1,
  caption: "Graph displaying average time for 3 determined novel views across 3 runs. ",
  height: 15%,
)


#let all-views-figure = figure(
  stack(
    dir: ttb,
    spacing: 0.8em,
    image-figure(
      ("../images/GT_view1.png", [GT View 1]),
      ("../images/GT_view2.png", [GT View 2]),
      ("../images/GT_view3.png", [GT View 3]),
      ("../images/spectral_view1.png", [Spectral View 1]),
      ("../images/spectral_view2.png", [Spectral View 2]),
      ("../images/spectral_view3.png", [Spectral View 3]),
      ("../images/large_color_view1.png", [Large Color View 1]),
      ("../images/large_color_view2.png", [Large Color View 2]),
      ("../images/large_color_view3.png", [Large Color View 3]),
      columns: 9,
    ),
    // Use a custom grid just for the 6 items, with 1.5fr padding on both sides to perfectly center it underneath the 9 columns above
    grid(
      columns: (1.5fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1.5fr),
      gutter: 0.8em,
      [],
      // Empty block acting as spacer
      stack(dir: ttb, spacing: 0.4em, image("../images/test_larger_view1.png", width: 100%), align(center, text(
        size: 8pt,
        [Large T View 1],
      ))),
      stack(dir: ttb, spacing: 0.4em, image("../images/test_larger_view2.png", width: 100%), align(center, text(
        size: 8pt,
        [Large T View 2],
      ))),
      stack(dir: ttb, spacing: 0.4em, image("../images/test_larger_view3.png", width: 100%), align(center, text(
        size: 8pt,
        [Large T View 3],
      ))),
      stack(dir: ttb, spacing: 0.4em, image("../images/medium_tAfter_linear_T_view1.png", width: 100%), align(
        center,
        text(size: 8pt, [Medium View 1]),
      )),
      stack(dir: ttb, spacing: 0.4em, image("../images/medium_tAfter_linear_T_view2.png", width: 100%), align(
        center,
        text(size: 8pt, [Medium View 2]),
      )),
      stack(dir: ttb, spacing: 0.4em, image("../images/medium_tAfter_linear_T_view3.png", width: 100%), align(
        center,
        text(size: 8pt, [Medium View 3]),
      )),
      [],
      // Empty block acting as spacer
    ),
  ),
)

#all-views-figure

#timings-graph

#let memory-table = figure(
  {
    set text(size: 8pt)
    table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: horizon,
      table.header(
        [*Network Configuration*],
        [*Hashgrid Size*],
        [*MLP Size*],
        [*Staging Buffers* (e.g. `inferBefore`)],
        [tcnn activation memory],
        [Total],
      ),
      [
        Medium.json
      ],
      [
        /*12 $times$ 524,288 $times$ 2 $times$ 2 =*/ 25,165,824 bytes $approx$ 24MB
      ],
      [
        /*(3 $times 64^2$ + 64 $times$ 16) $times 2$ = */13,312 $times$ 2 = 26,624 bytes $approx$ 26KB
      ],
      [
        780,032 $times$ 188 = 146,646,016 bytes $approx$ 147 MB
      ],
      [
        (3 $times$ 64 + 48) $times$ 2 $times$ 780032 = 374,415,360 bytes $approx$ 374 MB (with 75 MB input staging buffer)
      ],
      [
        546,253,824 bytes $approx$ 546 MB
      ],

      [
        Large.json (both T and color)
      ],
      [
        33,554,432 bytes $approx$ 32 MB
      ],
      [
        /*64 * 128 + 128 * 128 * 2 + 128 * 16 = */43008 bytes $times$ 2 $approx$ 86KB
      ],
      [
        780,032 $times$ 188 = 146,646,016 bytes $approx$ 147 MB
      ],
      [
        (3 $times$ 128 $times$ + 64) 2 $times$ 780032 = 698,908,672 bytes $approx$ 699 MB (with 100MB input staging buffer)

      ],
      [
        879,195,136 bytes $approx$ 879MB
      ],

      [
        Spectral.json
      ],
      [
        33,554,432 bytes $approx$ 32 MB
      ],
      [
        /*64 * 128 + 128 * 128 * 2 + 128 * 16 = */43008 bytes $times$ 2 $approx$ 86KB
      ],
      [
        780,032 $times$ 188 = 146,646,016 bytes $approx$ 147 MB
      ],
      [
        (3 $times$ 128 $times$ + 64) 2 $times$ 780032 = 698,908,672 bytes $approx$ 699 MB (with 100MB input staging buffer)
      ],
      [
        879,195,136 bytes $approx$ 879 MB (less if optimized to exclude certain staging buffers)
      ],
    )
  },
  caption: [Table showing memory usage per configuration. Usage is split between Network weight size and Staging Buffer size with JIT Fusion disabled. If JIT Fusion is enabled, the tcnn staging memory drastically decreases and only accounts for input size],
)

#memory-table



#let flame_visualizations = image-figure(
  "../images/testing_images/original_failed_2_13_64_2_without_padding/flame_7.png",
  "../images/testing_images/original_failed_2_13_64_2_without_padding/flame_8.png",
  "../images/testing_images/original_failed_2_13_64_2_without_padding/flame_9.png",
  "../images/testing_images/original_failed_2_13_64_2_without_padding/flame_10.png",
  "../images/testing_images/original_failed_2_13_64_2_without_padding/flame_11.png",
  // "../images/testing_images/chosen_8_13_64_2_clamping_purple_msgpack/flame_7.png",
  // "../images/testing_images/chosen_8_13_64_2_clamping_purple_msgpack/flame_8.png",
  // "../images/testing_images/chosen_8_13_64_2_clamping_purple_msgpack/flame_9.png",
  // "../images/testing_images/chosen_8_13_64_2_clamping_purple_msgpack/flame_10.png",
  // "../images/testing_images/chosen_8_13_64_2_clamping_purple_msgpack/flame_11.png",
  "../images/radiance_imgs/medium_7.png",
  "../images/radiance_imgs/medium_8.png",
  "../images/radiance_imgs/medium_9.png",
  "../images/radiance_imgs/medium_10.png",
  "../images/radiance_imgs/medium_11.png",
  "../images/radiance_imgs/large_color_6.png",
  "../images/radiance_imgs/large_color_7.png",
  "../images/radiance_imgs/large_color_8.png",
  "../images/radiance_imgs/large_color_10.png",
  "../images/radiance_imgs/large_color_11.png",
  columns: 5,
  height: 2cm,
  caption: [Images for the the z-slice visualizations. Rows in order is: `validation_config`, `Medium`, `Large`],
)


#flame_visualizations



#let bunny_examples = image-figure(
  ("../images/bunny_orig_view1.png", [bunny original view 1]),
  ("../images/bunny_orig_view2.png", [bunny original view 2]),
  ("../images/bunny_orig_view3.png", [bunny original view 3]),
  ("../images/bunny_native_view1.png", [bunny GT view 1]),
  ("../images/bunny_native_view2.png", [bunny GT view 2]),
  ("../images/bunny_native_view3.png", [bunny GT view 3]),
  ("../images/bunny2_view1.png", [bunny GT view 1]),
  ("../images/bunny2_view2.png", [bunny GT view 2]),
  ("../images/bunny2_view3.png", [bunny GT view 3]),
  columns: 3,
)

#bunny_examples


#let plume_examples = image-figure(
  ("../images/smoke_native_orig_view1.png", [plume original view 1]),
  ("../images/smoke_native_orig_view2.png", [plume original view 2]),
  ("../images/smoke_native_orig_view3.png", [plume original view 3]),
  ("../images/smoke_gt_view1.png", [plume GT view 1]),
  ("../images/smoke_gt_view2.png", [plume GT view 2]),
  ("../images/smoke_gt_view3.png", [plume GT view 3]),
  ("../images/smoke_view1.png", [plume GT view 1]),
  ("../images/smoke_view2.png", [plume GT view 2]),
  ("../images/smoke_view3.png", [plume GT view 3]),
  columns: 3,
)

#plume_examples



#let green_T_slices = image-figure(
  ("../images/transmittance_imgs/green_0_T.png", [green_0_T]),
  ("../images/transmittance_imgs/green_1_T.png", [green_1_T]),
  ("../images/transmittance_imgs/green_2_T.png", [green_2_T]),
  ("../images/transmittance_imgs/green_3_T.png", [green_3_T]),
  ("../images/transmittance_imgs/green_4_T.png", [green_4_T]),
  ("../images/transmittance_imgs/green_5_T.png", [green_5_T]),
  ("../images/transmittance_imgs/green_6_T.png", [green_6_T]),
  ("../images/transmittance_imgs/green_7_T.png", [green_7_T]),
  ("../images/transmittance_imgs/green_8_T.png", [green_8_T]),
  ("../images/transmittance_imgs/green_9_T.png", [green_9_T]),
  ("../images/transmittance_imgs/green_10_T.png", [green_10_T]),
  ("../images/transmittance_imgs/green_11_T.png", [green_11_T]),
  ("../images/transmittance_imgs/green_12_T.png", [green_12_T]),
  ("../images/transmittance_imgs/green_13_T.png", [green_13_T]),
  ("../images/transmittance_imgs/green_14_T.png", [green_14_T]),
  ("../images/transmittance_imgs/green_15_T.png", [green_15_T]),
  ("../images/transmittance_imgs/green_16_T.png", [green_16_T]),
  ("../images/transmittance_imgs/green_17_T.png", [green_17_T]),
  ("../images/transmittance_imgs/green_18_T.png", [green_18_T]),
  ("../images/transmittance_imgs/green_19_T.png", [green_19_T]),

  columns: 5,
)


#green_T_slices


#let magenta_T_slices = image-figure(
  ("../images/transmittance_imgs/magenta_0_T.png", [magenta_0_T]),
  ("../images/transmittance_imgs/magenta_1_T.png", [magenta_1_T]),
  ("../images/transmittance_imgs/magenta_2_T.png", [magenta_2_T]),
  ("../images/transmittance_imgs/magenta_3_T.png", [magenta_3_T]),
  ("../images/transmittance_imgs/magenta_4_T.png", [magenta_4_T]),
  ("../images/transmittance_imgs/magenta_5_T.png", [magenta_5_T]),
  ("../images/transmittance_imgs/magenta_6_T.png", [magenta_6_T]),
  ("../images/transmittance_imgs/magenta_7_T.png", [magenta_7_T]),
  ("../images/transmittance_imgs/magenta_8_T.png", [magenta_8_T]),
  ("../images/transmittance_imgs/magenta_9_T.png", [magenta_9_T]),
  ("../images/transmittance_imgs/magenta_10_T.png", [magenta_10_T]),
  ("../images/transmittance_imgs/magenta_11_T.png", [magenta_11_T]),
  ("../images/transmittance_imgs/magenta_12_T.png", [magenta_12_T]),
  ("../images/transmittance_imgs/magenta_13_T.png", [magenta_13_T]),
  ("../images/transmittance_imgs/magenta_14_T.png", [magenta_14_T]),
  ("../images/transmittance_imgs/magenta_15_T.png", [magenta_15_T]),
  ("../images/transmittance_imgs/magenta_16_T.png", [magenta_16_T]),
  ("../images/transmittance_imgs/magenta_17_T.png", [magenta_17_T]),
  ("../images/transmittance_imgs/magenta_18_T.png", [magenta_18_T]),
  ("../images/transmittance_imgs/magenta_19_T.png", [magenta_19_T]),

  columns: 5,
)

#magenta_T_slices


#let medium_T_slices = image-figure(
  ("../images/transmittance_imgs/medium_0_T.png", [medium_0_T]),
  ("../images/transmittance_imgs/medium_1_T.png", [medium_1_T]),
  ("../images/transmittance_imgs/medium_2_T.png", [medium_2_T]),
  ("../images/transmittance_imgs/medium_3_T.png", [medium_3_T]),
  ("../images/transmittance_imgs/medium_4_T.png", [medium_4_T]),
  ("../images/transmittance_imgs/medium_5_T.png", [medium_5_T]),
  ("../images/transmittance_imgs/medium_6_T.png", [medium_6_T]),
  ("../images/transmittance_imgs/medium_7_T.png", [medium_7_T]),
  ("../images/transmittance_imgs/medium_8_T.png", [medium_8_T]),
  ("../images/transmittance_imgs/medium_9_T.png", [medium_9_T]),
  ("../images/transmittance_imgs/medium_10_T.png", [medium_10_T]),
  ("../images/transmittance_imgs/medium_11_T.png", [medium_11_T]),
  ("../images/transmittance_imgs/medium_12_T.png", [medium_12_T]),
  ("../images/transmittance_imgs/medium_13_T.png", [medium_13_T]),
  ("../images/transmittance_imgs/medium_14_T.png", [medium_14_T]),
  ("../images/transmittance_imgs/medium_15_T.png", [medium_15_T]),
  ("../images/transmittance_imgs/medium_16_T.png", [medium_16_T]),
  ("../images/transmittance_imgs/medium_17_T.png", [medium_17_T]),
  ("../images/transmittance_imgs/medium_18_T.png", [medium_18_T]),
  ("../images/transmittance_imgs/medium_19_T.png", [medium_19_T]),

  columns: 5,
)

#medium_T_slices


#let spectral_T_slices = image-figure(
  ("../images/spectral_T_imgs/spectral_0_T.png", [spectral_0_T]),
  ("../images/spectral_T_imgs/spectral_1_T.png", [spectral_1_T]),
  ("../images/spectral_T_imgs/spectral_2_T.png", [spectral_2_T]),
  ("../images/spectral_T_imgs/spectral_3_T.png", [spectral_3_T]),
  ("../images/spectral_T_imgs/spectral_4_T.png", [spectral_4_T]),
  ("../images/spectral_T_imgs/spectral_5_T.png", [spectral_5_T]),
  ("../images/spectral_T_imgs/spectral_6_T.png", [spectral_6_T]),
  ("../images/spectral_T_imgs/spectral_7_T.png", [spectral_7_T]),
  ("../images/spectral_T_imgs/spectral_8_T.png", [spectral_8_T]),
  ("../images/spectral_T_imgs/spectral_9_T.png", [spectral_9_T]),
  ("../images/spectral_T_imgs/spectral_10_T.png", [spectral_10_T]),
  ("../images/spectral_T_imgs/spectral_11_T.png", [spectral_11_T]),
  ("../images/spectral_T_imgs/spectral_12_T.png", [spectral_12_T]),
  ("../images/spectral_T_imgs/spectral_13_T.png", [spectral_13_T]),
  ("../images/spectral_T_imgs/spectral_14_T.png", [spectral_14_T]),
  ("../images/spectral_T_imgs/spectral_15_T.png", [spectral_15_T]),
  ("../images/spectral_T_imgs/spectral_16_T.png", [spectral_16_T]),
  ("../images/spectral_T_imgs/spectral_17_T.png", [spectral_17_T]),
  ("../images/spectral_T_imgs/spectral_18_T.png", [spectral_18_T]),
  ("../images/spectral_T_imgs/spectral_19_T.png", [spectral_19_T]),

  columns: 5,
)

#spectral_T_slices


#let large_T_slices = image-figure(
  ("../images/large_T_imgs/large_0_T.png", [large_0_T]),
  ("../images/large_T_imgs/large_1_T.png", [large_1_T]),
  ("../images/large_T_imgs/large_2_T.png", [large_2_T]),
  ("../images/large_T_imgs/large_3_T.png", [large_3_T]),
  ("../images/large_T_imgs/large_4_T.png", [large_4_T]),
  ("../images/large_T_imgs/large_5_T.png", [large_5_T]),
  ("../images/large_T_imgs/large_6_T.png", [large_6_T]),
  ("../images/large_T_imgs/large_7_T.png", [large_7_T]),
  ("../images/large_T_imgs/large_8_T.png", [large_8_T]),
  ("../images/large_T_imgs/large_9_T.png", [large_9_T]),
  ("../images/large_T_imgs/large_10_T.png", [large_10_T]),
  ("../images/large_T_imgs/large_11_T.png", [large_11_T]),
  ("../images/large_T_imgs/large_12_T.png", [large_12_T]),
  ("../images/large_T_imgs/large_13_T.png", [large_13_T]),
  ("../images/large_T_imgs/large_14_T.png", [large_14_T]),
  ("../images/large_T_imgs/large_15_T.png", [large_15_T]),
  ("../images/large_T_imgs/large_16_T.png", [large_16_T]),
  ("../images/large_T_imgs/large_17_T.png", [large_17_T]),
  ("../images/large_T_imgs/large_18_T.png", [large_18_T]),
  ("../images/large_T_imgs/large_19_T.png", [large_19_T]),

  columns: 5,
)

#large_T_slices
