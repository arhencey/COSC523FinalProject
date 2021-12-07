# List of potential libraries we might use 
library(dplyr) 
library(tidytext)
# install.packages("textfeatures")
library(textfeatures)
library(tidyverse)
library(tm)
library(stringr)

#Loading the training data set
excerpts <- read.csv("Assignment 3/commonlitreadabilityprize/train.csv") #Reading the train.csv file
View(excerpts)

#Processing the excerpts data and cleaning the text
#Removing unwanted content from Data frame like urls, special charcters etc
replace_reg <- "https://t.co/[A-Za-z\\d]+|http://[A-Za-z\\d]+|&amp;|&lt;|&gt;|RT|https"
tidy_excerpts <- excerpts %>% filter(!str_detect(excerpt, "^RT")) %>% mutate(excerpt = str_replace_all(excerpt, replace_reg, " ")) %>% mutate(excerpt = str_replace_all(excerpt, "[\r\n]", " "))
tidy_excerptstemp <- tidy_excerpts %>% mutate(excerpt = str_replace_all(excerpt, "[^[:alnum:]]", " "))%>% mutate(excerpt=iconv(excerpt, from = 'UTF-8', to = 'ASCII//TRANSLIT'))

#Designing a function for computing total Scrabble score
#Defining the dataframe containing letters and their corresponding letter score
letterscore<-read.csv(text="letter,score
A,1
B,3
C,3
D,2
E,1
F,4
G,2
H,4
I,1
J,8
K,5
L,1
M,3
N,1
O,1
P,3
Q,10
R,1
S,1
T,1
U,1
V,4
W,4
X,8
Y,4
Z,10
0,0
1,0
2,0
3,0
4,0
5,0
6,0
7,0
8,0
9,0",header=T)

rownames(letterscore)<-letterscore$letter

#Remove punctuation
removepunc<-tidy_excerptstemp %>% mutate(excerpt=removePunctuation(excerpt))

#Remove whitespaces
removespace <- removepunc %>% mutate(excerpt=str_replace_all(string=excerpt, pattern=" ", repl=""))

#Capitalize Letters
Capspace <- removespace %>% mutate(excerpt=str_to_upper(string=excerpt))

test<-strsplit(Capspace$excerpt, split = "")

#Calculating Scrabble score
lengthtest=length(test)
scrabblescore=integer(lengthtest)
for (j in 1:lengthtest){
  sum=integer(1)
  temp=length(test[[j]])
  for (i in 1:temp){
    sum=sum+as.numeric(letterscore[letterscore$letter==test[[j]][i],"score"])
  }
  scrabblescore[j]=sum
}

#Using textfeatures package in R in order to extract and add different text features for the excerpt data
#Changing excerpt heading to text for use in textfeatures
colnames(tidy_excerpts)[colnames(tidy_excerpts)=="excerpt"] <- "text"

#Get minimum and maximum characters in a word for each excerpt
#maximum characters
maxchars<-sapply(strsplit(tidy_excerpts$text, " "), function(x) nchar(x[which.max(nchar(x))]))
#minimum characters
minchars<-sapply(strsplit(tidy_excerpts$text, " "), function(x) nchar(x[which.min(nchar(x)<=3)]))

#Extra features
library(quanteda)
library(quanteda.textstats)

#Measuring text difficulty through different scores
difficscores<-textstat_readability(tidy_excerpts$text,measure = c("Flesch","Flesch.Kincaid","Spache","SMOG","Scrabble","FOG"))

#Unnormalized features (mainly used for visualization)
excerptfeat<-textfeatures(tidy_excerpts,sentiment=TRUE,word_dims = 0,normalize = FALSE)
excerptfeat1<-select(excerptfeat,-1:-6)

#Combining new generated features with the original data
tidy_excerpts1<-cbind(tidy_excerpts,excerptfeat1,scrabblescore,maxchars,minchars,difficscores[,c(2:7)])

#Remove the redundant features found
traintarget<-select(tidy_excerpts1,5)
trainfeatures<-tidy_excerpts1[-c(1:6,8:9,10:21,23:34,37)]
mldata<-cbind(trainfeatures,traintarget)
