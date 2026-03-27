//Show rules
#set page(margin: (left: 1cm, right: 1cm, top: 1cm, bottom: 1cm))
#set page(numbering: "1 of 1", number-align: right)
#show heading.where(level: 2): it => pad(left: 0.5em, it)
#show heading.where(level: 3): it => pad(left: 1em, it)
#show heading.where(level: 4): it => pad(left: 1.5em, it)

// Custom bindings
#let p(depth, body) = block(inset: (left: 0.5em * (depth - 1)), text(size: 9pt, body))
#let vb(val) = math.bold(math.upright(val))
#let image-grid(..paths, columns: 3, height: auto) = grid(
  columns: (1fr,) * columns,
  gutter: 1em,
  ..paths.pos().map(path => image(path, width: 100%, height: height))
)
#let image-figure(..images, columns: 3, caption: none, gap: 0.8em, height: auto, kind: auto, supplement: auto) = {
  // Each positional arg is either a path string or a (path, caption) tuple.
  let items = images
    .pos()
    .map(item => {
      if type(item) == str {
        (path: item, caption: none)
      } else {
        // Expect an array of (path, caption)
        (path: item.at(0), caption: item.at(1, default: none))
      }
    })
  figure(
    grid(
      columns: (1fr,) * columns,
      gutter: gap,
      ..items.map(it => {
        let img = if height == auto {
          image(it.path, width: 100%)
        } else {
          image(it.path, height: height, fit: "contain")
        }
        if it.caption != none {
          stack(dir: ttb, spacing: 0.4em, img, align(center, text(size: 8pt, it.caption)))
        } else {
          img
        }
      })
    ),
    caption: caption,
    kind: kind,
    supplement: supplement,
  )
}


