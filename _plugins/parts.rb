# A part page (tutorial.md, discussion.md, thoughts.md) is on the site only if its review is.
# Jekyll has no notion of a parent document, so without this a draft review's tutorial would be
# published on its own just because the part file forgot to say `published: false`.
Jekyll::Hooks.register :site, :post_read do |site|
  papers = site.collections["papers"]
  next unless papers

  reviews = papers.docs.reject { |d| d.data["part"] }.map { |d| d.data["slug"] }
  papers.docs.select! { |d| !d.data["part"] || reviews.include?(d.data["slug"]) }
end
