'''
35 trials
bm25 without filtering out age and gender
[1, 1, 0, 0, 0, 2, 0, 2, 1, 0] 0.675
bm25+sciSPacy (exc) without filtering out age and gender 
[2, 1, 0, 1, 2, 0, 0, 1, 1, 0] 0.84
bm25+sciSPacy (inc+exc) without filtering out age and gender 
[2, 1, 0, 1, 0, 1, 0, 0, 0, 0] 0.94
bm25+sciSPacy (tdm+inc+exc) without filtering out age and gender 
[2, 1, 0, 1, 2, 0, 0, 0, 0, 0] 0.85
bm25 with filtering out age and gender
[1, 0, 0, 2, 0, 2, 1, 0, 0, 0] 0.61
tfidf without filtering
[0, 0, 2, 0, 0, 0, 0, 1, 2, 0] 0.48
tfidf after filtering
[0, 2, 0, 0, 0, 1, 2, 0, 1, 1] 0.668
BERT with/without filtering
[1, 0, 0, 2, 0, 1, 0, 0, 0, 1] 0.65
'''

'''
50 trials
bm25 without filtering out age and gender
[1, 1, 2, 2, 0, 2, 0, 1, 0, 0] 0.816
bm25+sciSPacy (exc) without filtering out age and gender 
[2, 1, 0, 0, 1, 1, 0, 0, 0, 0] 0.924
bm25+sciSPacy (inc+exc) without filtering out age and gender 
[1, 2, 0, 0, 0, 1, 0, 0, 0, 0] 0.93
bm25+sciSPacy (tdm+inc+exc) without filtering out age and gender 
[0, 0, 2, 0, 2, 0, 1, 2, 0, 0] 0.546
bm25 with filtering out age and gender
[1, 1, 2, 2, 0, 2, 0, 0, 0, 0] 0.813
tfidf without filtering
[0, 1, 1, 2, 2, 2, 0, 1, 1, 0] 0.708
tfidf after filtering
[0, 1, 2, 2, 2, 0, 1, 1, 0, 0] 0.73
BERT with/without filtering
[1, 0, 0, 2, 0, 1, 0, 0, 0, 0] 0.657
'''

'''
brief summary + 50 trials

bm25 without filtering out age and gender
[2, 1, 2, 0, 0, 0, 0, 0, 0, 0] 0.92
bm25+sciSPacy (exc) without filtering out age and gender 
[1, 2, 0, 1, 0, 2, 0, 0, 2, 0] 0.79
bm25+sciSPacy (inc+exc) without filtering out age and gender 
[0, 0, 0, 0, 2, 0, 2, 0, 1, 0] 0.407
bm25+sciSPacy (tdm+inc+exc) without filtering out age and gender 
[0, 0, 2, 0, 2, 2, 0, 2, 0, 0] 0.57
bm25 with filtering out age and gender
[2, 1, 2, 0, 0, 0, 0, 1, 2, 0] 0.84
tfidf without filtering
[2, 1, 0, 0, 2, 1, 1, 0, 1, 2] 0.796
tfidf after filtering
[2, 0, 2, 1, 1, 0, 1, 2, 0, 0] 0.79
BERT
[0, 2, 1, 0, 0, 1, 2, 2, 0, 1] 0.71
'''

'''
brief summary + 700 trials

bm25 without filtering out age and gender
[1, 0, 0, 0, 0, 1, 0, 0, 0, 0] 0.832
bm25+sciSPacy (exc) without filtering out age and gender 
[1, 0, 0, 0, 0, 0, 0, 0, 1, 2] 0.525
bm25+sciSPacy (inc+exc) without filtering out age and gender 
[0, 1, 0, 2, 0, 0, 0, 0, 0, 0] 0.53
bm25+sciSPacy (tdm+inc+exc) without filtering out age and gender 
[0, 0, 1, 0, 1, 0, 0, 0, 1, 0] 0.557
bm25 with filtering out age and gender
[1, 0, 1, 0, 0, 0, 2, 0, 0, 0] 0.605
tfidf without filtering
[1, 0, 1, 2, 0, 0, 0, 0, 0, 0] 0.676
tfidf after filtering
[1, 0, 2, 0, 0, 0, 1, 0, 0, 1] 0.684
BERT
[2, 2, 0, 0, 0, 0, 0, 0, 1, 0] 0.963
'''

'''
brief summary + description + 700 trials

bm25 without filtering out age and gender
[1, 0, 0, 0, 1, 1, 0, 0, 0, 0] 0.818
bm25+sciSPacy (exc) without filtering out age and gender 

bm25+sciSPacy (inc+exc) without filtering out age and gender 
[1, 2, 0, 2, 0, 0, 0, 0, 1, 1] 0.769
bm25+sciSPacy (tdm+inc+exc) without filtering out age and gender 

bm25 with filtering out age and gender
[1, 0, 0, 1, 1, 0, 0, 0, 2, 0] 0.596
tfidf without filtering
[1, 0, 1, 0, 0, 0, 0, 2, 1, 1] 0.614
tfidf after filtering
[1, 0, 0, 0, 2, 1, 1, 0, 0, 0] 0.625
BERT
[0, 2, 2, 0, 0, 0, 0, 1, 0, 0] 0.688
'''

'''
bm25 without filtering with different weights:

.23 .33 .44 [0, 1, 1, 0, 0, 0, 2, 0, 2, 0] 0.521

.33 .44 .44 [1, 0, 0, 0, 0, 1, 0, 0, 2, 1] 0.559
.33 .44 .33 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0] 0.832
.33 .44 .23 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0] 0.832
.33 .44 .15 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0] 0.832
.33 .44 .10 [1, 0, 0, 0, 0, 0, 1, 0, 1, 0] 0.767
.33 .44 .05 [1, 0, 0, 0, 0, 0, 1, 0, 1, 0] 0.767
.33 .44 .01 [1, 0, 0, 0, 0, 1, 0, 0, 0, 1] 0.772
.33 .44   0 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0] 0.832
'''
