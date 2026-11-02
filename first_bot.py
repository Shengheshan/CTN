import capturethenarrative as CTN

social = CTN.legitsocial.LegitSocial()

# 1. 获取当前的热门帖子

print("== Trending Posts ==")

for post in social.feed_trending(limit=15):
    print(post)

# 2.获取当前新闻

news = CTN.newssites.NewsSites()

print("== Current News ==")

for article in news.latest(5):
    print(article)

