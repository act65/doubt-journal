# The scoreboard (XP, rank, streaks, penalties) is for the editor, not the readers. A page with
# `scoreboard: true` in its front matter is only built when the site config sets `scoreboard:
# true`, which only _config.scoreboard.yml does:
#
#   bundle exec jekyll serve --config _config.yml,_config.scoreboard.yml
#
# Templates hide their scoreboard bits with `{% if site.scoreboard %}`.
Jekyll::Hooks.register :site, :post_read do |site|
  next if site.config["scoreboard"]

  site.pages.reject! { |p| p.data["scoreboard"] }
end
