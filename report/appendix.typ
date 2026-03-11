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
  ("../images/even_even_larger_color_view1.png", [View 1]),
  ("../images/even_even_larger_color_view2.png", [View 2]),
  ("../images/even_even_larger_color_view3.png", [View 3]),
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
  ("../images/good_12_19_1-38_tAfter_linear_T_view1.png", [View 1]),
  ("../images/good_12_19_1-38_tAfter_linear_T_view2.png", [View 2]),
  ("../images/good_12_19_1-38_tAfter_linear_T_view3.png", [View 3]),
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
      ("../images/even_even_larger_color_view1.png", [Large Color View 1]),
      ("../images/even_even_larger_color_view2.png", [Large Color View 2]),
      ("../images/even_even_larger_color_view3.png", [Large Color View 3]),
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
      stack(dir: ttb, spacing: 0.4em, image("../images/good_12_19_1-38_tAfter_linear_T_view1.png", width: 100%), align(
        center,
        text(size: 8pt, [Medium View 1]),
      )),
      stack(dir: ttb, spacing: 0.4em, image("../images/good_12_19_1-38_tAfter_linear_T_view2.png", width: 100%), align(
        center,
        text(size: 8pt, [Medium View 2]),
      )),
      stack(dir: ttb, spacing: 0.4em, image("../images/good_12_19_1-38_tAfter_linear_T_view3.png", width: 100%), align(
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
  "../images/radiance_imgs/good_12_19_1-38_tAfter_linear_7.png",
  "../images/radiance_imgs/good_12_19_1-38_tAfter_linear_8.png",
  "../images/radiance_imgs/good_12_19_1-38_tAfter_linear_9.png",
  "../images/radiance_imgs/good_12_19_1-38_tAfter_linear_10.png",
  "../images/radiance_imgs/good_12_19_1-38_tAfter_linear_11.png",
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
