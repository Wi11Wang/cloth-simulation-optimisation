#let project(title: "", author: "", body) = {
  // Set the document's basic properties.
  set document(author: author, title: title)
  set text(font: "New Computer Modern", lang: "en")

  let time = datetime.today().display("[month repr:long] [day], [year]")

  // Main body.
  set page(
    numbering: "1", 
    number-align: center,
    header: locate(
        loc => if [#loc.page()] == [1] {
            []
        } else {
            [#title #h(1fr) #author]
        }
    )
  )
  // Title row.
  align(center)[
    #pad(
      bottom: 1em,
      block(text(weight: 600, 2em, title))
    )
    #pad(
      bottom: 1em,
      block(text(weight: 400, 1.2em, author))
    )
  ]
  // show par: set block(spacing: 0.65em)
  set par(
    first-line-indent: 1em,
    justify: true,
  )

  counter(page).update(1)
  body
}

#let result_table(table_prefix, caption, placement: none) = {
  let table_0 = csv(table_prefix + "_0.csv", delimiter: ";")
  let table_1 = csv(table_prefix + "_1.csv", delimiter: ";")
  figure(
    placement: placement,
    grid(
      rows: 2,
    )[
      #table(
        columns: (1fr, 1fr, 1fr, 1fr, 1fr),
        align: center,
        row-gutter: (3pt, 0pt, 0pt, 2pt, 0pt, 0pt, 2pt, 0pt, 0pt, 2pt, 0pt, 0pt, 2pt, 0pt),
        [*$n$*], [*$d$*], [*wall time ($mu s$)*], [*`PAPI_DP_OPS`*], [*`MFLOPS`*], 
        ..table_0.flatten(),
      )
      #table(
        columns: (1fr, 1fr, 1fr, 1fr, 1fr),
        align: center,
        row-gutter: (3pt, 0pt, 0pt, 2pt, 0pt, 0pt, 2pt, 0pt, 0pt, 2pt, 0pt, 0pt, 2pt, 0pt),
        [*`PAPI_L1_DCM`*], [*`PAPI_L2_DCM`*], [*`PAPI_TOT_INS`*], [*`PAPI_BR_MSP`*], [*`PAPI_VEC_DP`*],
        ..table_1.flatten(),
      )
    ],
    caption: caption
  ) 
}

#let result_table_omp(table_prefix, caption) = {
  let table_0 = csv(table_prefix + "_0.csv", delimiter: ";")
  let table_1 = csv(table_prefix + "_1.csv", delimiter: ";")
  table(
    columns: (0.3fr, 0.66fr, 0.66fr, 0.67fr, 1fr, 1fr, 1fr),
    align: center,
    row-gutter: (4pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 2pt, 0pt),
    [`idx`], [*`n`*], [*`d`*], [*`p`*], [*wall time ($mu s$)*], [*`PAPI_DP_OPS`*], [*`MFLOPS`*], 
    ..table_0.flatten(),
  )
  table(
    columns: (0.3fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: center,
    row-gutter: (4pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 1.5pt, 0pt, 0pt, 0pt, 2pt, 0pt),
    [`idx`], [*`PAPI_L1_DCM`*], [*`PAPI_L2_DCM`*], [*`PAPI_TOT_INS`*], [*`PAPI_BR_MSP`*], [*`PAPI_VEC_DP`*],
    ..table_1.flatten(),
  )
  v(-14pt)
  figure(
    table(),
    caption: caption
  ) 
}

#let compare(before, after) = {
figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    block(
      fill: luma(240),
      width: 100%,
      inset: 8pt,
      radius: 4pt,
      [
        #set align(center)
        `before`
        #set align(left)
        #before
      ]
    ),
    block(
      fill: luma(240),
      width: 100%,
      inset: 8pt,
      radius: 4pt,
      [
        #set align(center)
        `after`
        #set align(left)
        #after
      ]
      ),
    ),
    kind: "comparison",
    supplement: [Comparison],
)
}

#let comparev(before, after, placement: none) = {
figure(
  placement: placement,
  grid(
    rows: auto,
    gutter: 10pt,
    block(
      fill: luma(240),
      width: 100%,
      inset: 8pt,
      radius: 4pt,
      [
        #set align(center)
        `before`
        #set align(left)
        #before
      ]
    ),
    block(
      fill: luma(240),
      width: 100%,
      inset: 8pt,
      radius: 4pt,
      [
        #set align(center)
        `after`
        #set align(left)
        #after
      ]
      ),
    ),
    kind: "comparison",
    supplement: [Comparison],
)
}

#let question(description) = {
  block(
    width: 100%,
    inset: 8pt,
    radius: 4pt,
    stroke: (paint: blue, dash: "dashed"),
    text(description, weight: "bold")
  )
}

#let warning(str) = {
  block(
    width: 100%,
    fill: red,
    inset: 4pt,
    align(
      center,
      text(str, fill: white, weight: "bold", size: 13pt)
    )
  )
}