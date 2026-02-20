//Show rules
#set page(margin: (left: 1cm, right: 1cm, top: 1cm, bottom: 1cm))
#set page(numbering: "1 of 1", number-align: right)
#show heading.where(level: 2): it => pad(left: 0.5em, it)
#show heading.where(level: 3): it => pad(left: 1em, it)
#show heading.where(level: 4): it => pad(left: 1.5em, it)

// Custom bindings
#let p(depth, body) = block(inset: (left: 0.5em * (depth - 1)), text(size: 9.5pt, body))
#let vb(val) = math.bold(math.upright(val))
#let image-grid(..paths, columns: 3) = grid(
  columns: (1fr,) * columns,
  gutter: 1em,
  ..paths.pos().map(path => image(path, width: 100%))
)
