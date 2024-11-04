# current data
training stage, do a `operation seperation` model to apply to those unclearly labeled dataset
* first, concate these truely marked ecog epochs and train model on them and see the seperation performance
* then, apply to unclearly marked data epoch to seperate and see if the timing do not conflicat with rough timing describtion given by tiantan.
# proposal to do `ecog generation`
the novalty here is that, the ecog we generate is not a one-channel temporal vector, but a `multi-channel ecog-array`
* usual eeg generation is to generate pattern related one-channel temporal vector
* we think of the surrounding area which is different but still related to the direct-pattern-related function area, and we will also genrate their ecog, considering the relative location of their location to the exact function area 
* as for the `validation of generation`, is to prepare a classification model, and see if the seperability raise while generated ecog merged into the real ecog.
# new idea
the ads of dataset tiantan can provide is that some sensors directly located in validated functional area, regarding of sth with application promise, we proposed a concept: `ecog aided brain navigation`
* the concept is to `use ecog to correct and concret the raw brief functional map on brain`
* by an average brain functional map, the doctor can know rough areas of different function (so that they can monitor the state of patient from it) after specify at least two functional point, so they do `neuron navigation operation`, but the accurancy of this functional map impression is not so clear.  
* the 6-sensors-in-line has 1 or 2 of specific function, we want to know the exact monitored function of other sensors