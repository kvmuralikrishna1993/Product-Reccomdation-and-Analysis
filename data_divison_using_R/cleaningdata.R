setwd("/Users/Muralikrishna/Desktop/")
setwd("review")
getwd()
sony = read.csv("sony.csv")
str(sony)

#-------------------------------
# For small dataset
#------------------------------
table(sony$Product.Name)
sort(table(sony$Product.Name))
reviews = subset(sony, Product.Name == "Sony Xperia E C1504 Unlocked Android Phone--U.S. Warranty (Pink)")
View(reviews)
#str(reviews)
reviews$X = NULL
reviews$Product.Name = NULL
reviews$Rating = NULL
View(reviews)
str(reviews)

write.csv(reviews, file = "gadjet.csv")
#------------------------


#install librarys

install.packages("tm")
library(tm)
#combining all reviews
review_text = paste(reviews$Reviews, collapse = " ")
#viewing all reviews at a time
#View(review_text)
#creating vectorsource for corpus (storing in temporary object in R) 
review_s <- VectorSource(review_text)

#passing vector source to corpus function(to keep as one document)
corpus <- Corpus(review_s)
View(corpus)
inspect(corpus)

#vcorpus <- VCorpus(review_s)
#inspect(vcorpus)


#----------------------
#attributes(corpus)
#attributes(vcorpus)
#attr(vcorpus,"class")
#vcorpus$content
#----------------------
#vcorpus <- tm_map(vcorpus, content_transformer(tolower))
#vinspect(corpus)

#converting all char to lower
corpus <- tm_map(corpus, content_transformer(tolower))
inspect(corpus)

corpus <- tm_map(corpus,removeNumbers)
inspect(corpus)

#removing punctuation
corpus <- tm_map(corpus,removePunctuation)
inspect(corpus)

corpus <- tm_map(corpus,stripWhitespace)
inspect(corpus)

corpus <- tm_map(corpus,removeWords,stopwords("english"))
inspect(corpus)



writeLines(as.character(corpus), con="corpus.txt")
dtm <- DocumentTermMatrix(corpus)
inspect(dtm)
View(dtm)


dtm2 <- as.matrix(dtm)
View(dtm)
#inspect(dtm2) <--- will not work

freq <- colSums(dtm2)
View(freq)

#sorted frequency

freq <- sort(freq, decreasing = TRUE)
View(freq)
head(freq)

install.packages('wordcloud')
library(wordcloud)

str(freq)

words <- names(freq)
wordcloud(words[1:100],freq[1:200])

stopwords("en")

sent = read.csv("sent.csv")
str(sent)
sentiment.polarity
colnames(sent)
names(sent)[names(sent) == "X0"] <- "polarity"
names(sent)[names(sent) == "X.switchfoot.http...twitpic.com.2y1zl...Awww..that.s.a.bummer...You.shoulda.got.David.Carr.of.Third.Day.to.do.it...D"] <- "tweet"

install.packages("tidyverse")
library(tidyverse)
sentiment <- sent %>% select(polarity, tweet)
str(sentiment)
sentiment = subset(sentiment, sentiment$polarity == 0 || sentiment$polarity== 4)
write.csv(sentiment, file = "sentiment.csv")
View(sentiment)
subset(sentiment, sentiment$polarity == 2)

#spliting data into test and train
setwd("/Users/Muralikrishna/Desktop/")
setwd("review")
getwd()
sentiment = read.csv("sentiment.csv")
sentiment <- sentiment %>% select(tweet,polarity)
str(sentiment)
sentiment$polarity <- as.character(sentiment$polarity)
str(sentiment)

install.packages("plyr")
library(plyr)
sentiment$polarity <- revalue(sentiment$polarity, c("0"="neg"))
sentiment$polarity <- revalue(sentiment$polarity, c("4"="pos"))
str(sentiment)

set.seed(1231)
#train_ind <- sample(seq_len(nrow(sentiment)), size = smp_size)
sample <- sample.int(n = nrow(sentiment), size = floor(.75*nrow(sentiment)), replace = F)

train <- sentiment[sample, ]
test  <- sentiment[-sample, ]

#train <- sentiment[train_ind, ]
#test <- sentiment[-train_ind, ]

str(train)
str(test)

write.csv(train, file = "sentiment_train.csv")
write.csv(test, file = "sentiment_test.csv")

View(sentiment)

pos <- subset(sentiment, sentiment$polarity == "pos")
View(pos)
str(pos)

neg <- subset(sentiment, sentiment$polarity == "neg")
View(neg)
str(neg)

library(plyr)
set.seed(1231)
pos_sample <- sample.int(n = nrow(pos), size = floor(.00625*nrow(pos)), replace = F)
positive <- pos[pos_sample, ]
str(positive) #sample size =5000
neg_sample <- sample.int(n = nrow(neg), size = floor(.00625*nrow(neg)), replace = F)
negative <- neg[neg_sample, ]
str(negative)#sample size =5000
positive$X = NULL
negative$X = NULL

library(dplyr)
dataset <- dplyr::bind_rows(positive,negative) #binding two sets
str(dataset)
View (dataset)
#taking random samples
set.seed(139)
sample <- sample.int(n = nrow(dataset), size = floor(.75*nrow(dataset)), replace = F)
train <- dataset[sample, ]
test  <- dataset[-sample, ]
str(test)
str(subset(train, train$polarity == "pos")) #checking positive samples = 3743
str(subset(train, train$polarity == "neg"))#checking negative samples = 3756
write.csv(train, file = "sentiment_train.csv")
write.csv(test, file = "sentiment_test.csv")


setwd("/Users/Muralikrishna/Desktop/")
setwd("review")
getwd()
ds = read.csv("sentiment_train.csv")
str(ds)
library(plyr)
set.seed(123)
sample <- sample.int(n = nrow(ds), size = floor(.80*nrow(ds)), replace = F)
train <- ds[sample, ]
test  <- ds[-sample, ]
str(train)
str(test)
write.csv(train, file = "sentiment_train.csv")
write.csv(test, file = "sentiment_test.csv")
