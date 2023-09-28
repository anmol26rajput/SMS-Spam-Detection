# SMS-Spam-Detection
SMS Spam Detection Project using python
The file we are going to use contains a collection of more than 5 thousand SMS phone messages. Using labeled ham and spam examples, we'll **train a machine learning model to learn to discriminate between ham/spam automatically**. Then, with a trained model, we'll be able to **classify arbitrary unlabeled messages** as ham or spam.

Here I  am going to develop an SMS spam detector using **SciKit Learn's Naive Bayes classifier algorithm**. However before feeding data to Machine Learning NB algorithim, we need to process each SMS with the help of Natural Language libraries.

# Summary
Let me give you a brief idea that I am going to follow in this notebook to create the model:


* First try to understand the data and its distribution with basic EDA with the help of Pandas and Matplotlib libraries. Also, check for any outliers by analysing the distribution graphs. 

* Now with the help of NLP library **"NLTK"**, first **remove the punctuation** and **special symbols** from all the SMS and then **lower case** them. You can even **tokenize** each SMS into sentences and words after removing punctuation & special symbols. Here I am just splitting each SMS into words with white spaces. However, tokenization and parsing may be the best idea to split the texts. Please note that converting all the data to lower case helps in the process of preprocessing and in later stages in the NLP application.

* Then remove the **Stopswords** from all the SMS.

* After processing each SMS, we will create the **WordCloud** for Spam and Ham messages for the visual representation of widely used words in both Spam and Ham messages.

* Now we can normalize the text by NLTK **lemmatization** or **stemming** or distinguishing by **part of speech (POC)**. However, sometimes these methods don't work well especially for text-messages due to the way a lot of people tend to use abbreviations or shorthand in SMS. E.g. "IDK" for "I don't know" or "wut" for "what". So we will not process the text by these methods.

* For now, we will have the messages as lists of tokens and now we need to convert each of these messages into a vector so that SciKit Learn's algorithm models can work with.

    We'll do that in three steps using the **bag-of-words (BOW)** model:
    

    * Count how many times does a word occur in each message (Known as term frequency - **TF**)
    
    * Weigh the counts, so that frequent tokens get lower weight (inverse document frequency - **IDF**)

# Natural Language Processing (NLP)

Here the messages are in the human-readable language which computer can't understand, so we have to use the NLP to make it possible for computers to read human (natural) language SMS and determine which parts are important.

So, Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.

NLP makes it possible for computers to read the text, hear speech, interpret it, measure sentiment and determine which parts are important.

The input and output of an NLP system can be −
* Speech
* Written Text

Learn **basics of NLP** from [my GitHub code](https://github.com/dktalaicha/Natural-Language-Processing/blob/master/Natural-Language-Processing-Basics.ipynb). This will help to understand the jargons of NLP.
    
    * Normalize the vectors to unit length, to abstract from the original text length (L2 norm)
    


* Once the messages represented as vectors, we can finally train our spam/ham classifier. Now we can actually use almost any sort of **classification algorithms** like Random Forest, Naive Bayes etc.
