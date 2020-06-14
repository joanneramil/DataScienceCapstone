# The following code was used to create a text prediction model

# Setting up the environment
library(tm)
library(tokenizers)
library(dplyr)
library(ggplot2)
library(caret)
library(stringr)
library(quanteda)
library(NLP)
library(rJava)
library(RWeka)
library(openNLP)
setwd("C:/Users/joann/Desktop/Data_Science_Coursera/datasciencecoursera/10. Capstone Project/DataScienceCapstone")

# CREATING BASE PREDICTION MODEL

# Read data in

dtTwit <- readLines(
    "final/en_US/en_US.twitter.txt", encoding = "UTF-8",
    skipNul = TRUE
)
dtBlogs <- readLines(
    "final/en_US/en_US.blogs.txt", encoding = "UTF-8"
)
dtNews <- readLines(
    "final/en_US/en_US.news_alt.txt", encoding = "UTF-8"
)

# Save sample indices and sample
set.seed(5171)
sampleTwitIdx <- sample(length(dtTwit), size = 99000)
sampleBlogsIdx <- sample(length(dtBlogs), size = 30000)
sampleNewsIdx <- sample(length(dtNews), size = 34500)

sampleTwit <- dtTwit[sampleTwitIdx]
sampleBlogs <- dtBlogs[sampleBlogsIdx]
sampleNews <- dtNews[sampleNewsIdx]

# Remove everthing but text and whitespace
sampleTwit <- gsub("[^a-zA-Z\' ]", " ", sampleTwit)
sampleTwit <- gsub("\'", "", sampleTwit)
sampleBlogs <- gsub("[^a-zA-Z\' ]", " ", sampleBlogs)
sampleBlogs <- gsub("\'", "", sampleBlogs)
sampleNews <- gsub("[^a-zA-Z\' ]", " ", sampleNews)
sampleNews <- gsub("\'", "", sampleNews)

# Create corpus object
library(tm)
docAll <- c(sampleBlogs, sampleNews, sampleTwit)
corAll <- VCorpus(VectorSource(docAll))

# Clean corpus object
cleanCorpus <- function(corp) {
    cc <- tm_map(corp, stripWhitespace)
    cc <- tm_map(cc, PlainTextDocument)
    cc <- tm_map(cc, content_transformer(tolower))
    return(cc)
}

corAll <- cleanCorpus(corAll)

# Create tokenizers
library(RWeka)
biTok <- function(x) NGramTokenizer(x, Weka_control(
    min=2, max=2))
triTok <- function(x) NGramTokenizer(x, Weka_control(
    min=3, max=3))
quadTok <- function(x) NGramTokenizer(x, Weka_control(
    min=4, max=4))

# Build TDMs
tdm2 <- TermDocumentMatrix(corAll, control=list(
    tokenize=biTok))
tdm3 <- TermDocumentMatrix(corAll, control=list(
    tokenize=triTok))
tdm4 <- TermDocumentMatrix(corAll, control=list(
    tokenize=quadTok))

# Remove (only) VERY sparse terms
tdm2 <- removeSparseTerms(tdm2, 0.99999)
tdm3 <- removeSparseTerms(tdm3, 0.99999)
tdm4 <- removeSparseTerms(tdm4, 0.99999)

# Create ordered term lists
bi_grams <- sort(slam::row_sums(tdm2, na.rm=T), 
                 decreasing=TRUE)
tri_grams <- sort(slam::row_sums(tdm3, na.rm=T), 
                  decreasing=TRUE)
quad_grams <- sort(slam::row_sums(tdm4, na.rm=T), 
                   decreasing=TRUE)

# Shorten lists
Shorten <- function (freqList, ngram_len) {
    obtained <- c()
    short <- c()
    for (i in seq(length(freqList))) {
        gram <- unlist(strsplit(names(freqList[i]), 
                                split = " "))
        if (!(paste(gram[1:(ngram_len - 1)], 
                    collapse = " ") %in% obtained)) {
            short <- c(short, freqList[i])
            obtained <- c(obtained, 
                          paste(gram[1:(ngram_len - 1)], 
                                collapse = " "))
        }
    }
    return(short)
}

bi_grams <- Shorten(bi_grams, 2)
tri_grams <- Shorten(tri_grams, 3)
quad_grams <- Shorten(quad_grams, 4)

# Create prediction function
ngram_pred <- function (words) {
    # separate string into individual words
    words <- gsub("[^a-zA-Z\' ]", " ", words)
    words <- gsub("\'", "", words)
    words <- gsub(" +", " ", words)
    words <- tolower(words)
    words <- unlist(strsplit(words, split = " "))
    
    ### GET WORD FROM BEST 4-GRAM ###
    if (length(words) > 2) {
        
        # create appropriate input
        input <- paste(
            words[(length(words)-2):length(words)], 
            collapse = " ")
        # search quad_grams for matching 4-gram
        quad_match <- quad_grams[grep(paste0(
            "^", input, " "), names(quad_grams))]
        # extract and return resulting prediction
        pred_word <- unlist(
            strsplit(names(quad_match), split = " "))[4]
        if (!is.null(pred_word)) {return(pred_word)}
        
    }
    
    ### GET WORD FROM BEST 3-GRAM ###
    if (length(words) > 1) {
        
        # create appropriate input
        input <- paste(words[length(words)-1], 
                       words[length(words)])
        # search tri_grams for matching 3-gram
        tri_match <- tri_grams[grep(paste0(
            "^", input, " "), names(tri_grams))]
        # extract and return resulting prediction
        pred_word <- unlist(
            strsplit(names(tri_match), split = " "))[3]
        if (!is.null(pred_word)) {return(pred_word)}
    }
    
    ### GET WORD FROM BEST 2-GRAM ###
    if (length(words) > 0) {
        
        # create appropriate input
        input <- words[length(words)]
        # search bi_grams for matching 2-gram
        bi_match <- bi_grams[grep(paste0(
            "^", input, " "), names(bi_grams))]
        # extract and return resulting prediction
        pred_word <- unlist(strsplit(names(bi_match), split = " "))[2]
        if (!is.null(pred_word)) {return(pred_word)}
    } 
    return("the")
}

# MEASURING MODEL ACCURACY

# Create test set and sample set
test_set <- c(dtTwit[-sampleTwitIdx], 
              dtBlogs[-sampleBlogsIdx], 
              dtNews[-sampleNewsIdx])

set.seed(6178)
test_set_sample <- sample(test_set, size = 500)

# Make vectors of sentences and remove everything but
# text and whitespace
test_set_sample <- unlist(strsplit(test_set_sample, 
                                   split = "[!?\\.]"))
test_set_sample <- gsub("[!?\\.]", "", test_set_sample)
test_set_sample <- test_set_sample[nchar(
    test_set_sample) > 1]

test_set_sample <- gsub("[^a-zA-Z\' ]", " ", test_set_sample)
test_set_sample <- gsub("\'", "", test_set_sample)
test_set_sample <- gsub(" +", " ", test_set_sample)
test_set_sample <- gsub("^ | $", "", test_set_sample)
test_set_sample <- test_set_sample[nchar(
    test_set_sample) > 1]

# Make list of vectors with words separated
test_set_sample_words <- sapply(test_set_sample, 
                                strsplit, 
                                split = " ", 
                                USE.NAMES = FALSE)

# Make matrix one row for each sentence and ncol equal
# to length of longest sentence
max_length <- max(sapply(test_set_sample_words, length))
for (i in seq(length(test_set_sample_words))) {
    while (length(test_set_sample_words[[i]]) < max_length) {
        test_set_sample_words[[i]] <- c(
            test_set_sample_words[[i]], 
            NA)
    }
}
test_set_sample_words <- do.call(rbind, test_set_sample_words)


# Create new matrix with test set word matrix and predicted
# next word at input, n, equals 1 to max length - 1.

gen_preds <- function (sent_mat) {
    new_mat <- sent_mat
    for (col_idx in seq(dim(sent_mat)[2] - 1)) {
        preds <- c()
        for (row_idx in seq(dim(sent_mat)[1])) {
            if (!is.na(sent_mat[row_idx, col_idx + 1])) {
                preds <- c(preds, ngram_pred(paste(
                    sent_mat[row_idx, 1:col_idx], 
                    collapse = " ")))
            } else {
                preds <- c(preds, NA)
            }
        }
        new_mat <- cbind(new_mat, preds)
        colnames(new_mat)[dim(new_mat)[2]] <- paste0(
            "pred", col_idx + 1)
    }
    return(new_mat)
}    

pred_mat <- gen_preds(test_set_sample_words)

# Create named vector with percent of correct predictions
# at each length of n from 1 until number of usable 
# sentences is less than 10.

get_pred_acc <- function (p_m) {
    acc_vec <- c()
    max_sent <- grep("^pred2$", colnames(p_m)) - 1
    for (col_act in seq(from = 2, to = max_sent)) {
        col_pred <- paste0("pred", col_act)
        if (sum(!(is.na(p_m[, col_pred]))) > 29) {
            predicted <- p_m[, col_pred]
            actual <- p_m[!is.na(predicted), col_act]
            predicted <- predicted[!is.na(predicted)]
            acc_vec <- c(acc_vec, mean(predicted == actual))
        }
    }
    return(acc_vec)
}

pred_acc_vec <- get_pred_acc(pred_mat)

###   TEST SPEED   ###

# Create matrix of running times for each sentence with
# n length inputs in each column

gen_pred_times <- function (sent_mat) {
    new_mat <- matrix(nrow = dim(sent_mat)[1], 
                      ncol = dim(sent_mat)[2] - 1)
    for (col_idx in seq(dim(sent_mat)[2] - 1)) {
        print(paste("col_idx:", col_idx))
        times <- c()
        for (row_idx in seq(dim(sent_mat)[1])) {
            if (!is.na(sent_mat[row_idx, col_idx + 1])) {
                ptm <- proc.time()[2]
                preds <- ngram_pred(paste(
                    sent_mat[row_idx, 1:col_idx], 
                    collapse = " "))
                times <- c(times, proc.time()[2] - ptm)
            } else {
                times <- c(times, NA)
            }
        }
        new_mat[, col_idx] <- times
    }
    return(new_mat)
}    

time_mat <- gen_pred_times(test_set_sample_words)


