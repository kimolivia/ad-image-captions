with open('twitter.pw', 'r') as f:
    twitter = f.read()

print(twitter, type(twitter), len(twitter))
t = twitter.split('\n')
print(t, len(t))