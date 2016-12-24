
# Word Prediction using Recurrent Neural Networks

Brian Burns  
Udacity Machine Learning Engineer Nanodegree
December 29, 2016

<!-- from https://review.udacity.com/#!/rubrics/108/view -->

## Definition

### Project Overview

<!-- Student provides a high-level overview of the project in layman's terms.
Background information such as the problem domain, the project origin, and
related data sets or input data is given. -->

$ x + y $

important in speech recognition to help resolve ambiguity.
until recently trigram was state of the art in word prediction.
can't use much bigger contexts than trigram because too many possibilities to store (calculate) and most counts would be zero.
have to back off to digrams when trigram counts are too small. 
eg if prompt is "dinosaur pizza", and you've never seen that pair before, must backoff to the bigram "pizza ___". 
but trigrams fail to use a lot of information that can be used to predict the next word.
doesn't understand similarities between words, eg cat and dog
so need to convert words into vector of syntactic and semantic features, and use the features to predict next word.
allows us to use much larger context, eg previous 10 words.
bengio pioneered this.
huge softmax layer
skipmax connections go straight from input to output words
was slightly worse than trigram
since then have been improved considerably




### Problem Statement

<!-- The problem which needs to be solved is clearly defined. A strategy for
solving the problem, including discussion of the expected solution, has been
made. -->

### Metrics

<!-- Metrics used to measure performance of a model or result are clearly
defined. Metrics are justified based on the characteristics of the problem. -->


## Analysis

### Data Exploration

<!-- If a dataset is present, features and calculated statistics relevant to the
problem have been reported and discussed, along with a sampling of the data. In
lieu of a dataset, a thorough description of the input space or input data has
been made. Abnormalities or characteristics about the data or input that need to
be addressed have been identified. -->



### Exploratory Visualization

<!-- A visualization has been provided that summarizes or extracts a relevant
characteristic or feature about the dataset or input data with thorough
discussion. Visual cues are clearly defined. -->


### Algorithms and Techniques

<!-- Algorithms and techniques used in the project are thoroughly discussed and
properly justified based on the characteristics of the problem. -->


### Benchmark

<!-- Student clearly defines a benchmark result or threshold for comparing
performances of solutions obtained. -->




## Methodology

### Data Preprocessing

<!-- All preprocessing steps have been clearly documented. Abnormalities or
characteristics about the data or input that needed to be addressed have been
corrected. If no data preprocessing is necessary, it has been clearly justified.
-->


raw text:

between their stems. The sky was now golden and the horizon, a horizon
of distant woods, it seemed, was purple.

But all that Dr. Ashton could find to say, after contemplating this
prospect for many minutes, was: "Abominable!"

A listener would have been aware, immediately upon this, of the sound
of footsteps coming somewhat hurriedly in the direction of the study:
by the resonance he could have told that they were traversing a much
larger room. Dr. Ashton turned round in his chair as the door opened,
and looked expectant. The incomer was a lady--a stout lady in the

split by sentence:

The sky was now golden and the horizon, a horizon of distant woods, it seemed, was purple.

But all that Dr. Ashton could find to say, after contemplating this prospect for many minutes, was: "Abominable!"

A listener would have been aware, immediately upon this, of the sound of footsteps coming somewhat hurriedly in the direction of the study: by the resonance he could have told that they were traversing a much larger room.

Dr. Ashton turned round in his chair as the door opened, and looked expectant.


### Implementation

<!-- The process for which metrics, algorithms, and techniques were implemented
with the given datasets or input data has been thoroughly documented.
Complications that occurred during the coding process are discussed. -->


### Refinement

<!-- The process of improving upon the algorithms and techniques used is clearly
documented. Both the initial and final solutions are reported, along with
intermediate solutions, if necessary. -->



## Results

### Model Evaluation and Validation

<!-- The final model's qualities - such as parameters - are evaluated in detail.
Some type of analysis is used to validate the robustness of the model's
solution. -->

### Justification

<!-- The final results are compared to the benchmark result or threshold with
some type of statistical analysis. Justification is made as to whether the final
model and solution is significant enough to have adequately solved the problem.
-->


## Conclusion

### Free-Form Visualization

<!-- A visualization has been provided that emphasizes an important quality
about the project with thorough discussion. Visual cues are clearly defined. -->

### Reflection

<!-- Student adequately summarizes the end-to-end problem solution and discusses
one or two particular aspects of the project they found interesting or
difficult. -->

### Improvement

<!-- Discussion is made as to how one aspect of the implementation could be
improved. Potential solutions resulting from these improvements are considered
and compared/contrasted to the current solution. -->


--------------------------------------------------------------------------------

## Quality

### Presentation

<!-- Project report follows a well-organized structure and would be readily
understood by its intended audience. Each section is written in a clear, concise
and specific manner. Few grammatical and spelling mistakes are present. All
resources used to complete the project are cited and referenced. -->

### Functionality

<!-- Code is formatted neatly with comments that effectively explain complex
implementations. Output produces similar results and solutions as to those
discussed in the project. -->






