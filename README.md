# Austin-Animal-Shelter

The Austin Animal Center is the largest no-kill animal shelter in the United States that provides care and shelter to over 18,000 animals each year and is involved in a range of county, city, and state-wide initiatives for the protection and care of abandoned, at-risk, and surrendered animals.

As part of the City of Austin Open Data Initiative, the Austin Animal Center makes available its collected dataset that contains statistics and outcomes of animals entering the Austin Animal Services system.

The dataset contains shelter outcomes of several types of animals and breeds from 10/1/2013 to the present with a hourly time frequency. The data is updated daily.

Column Description

age_upon_outcome        Age of the animal at the time at which it left the shelter.                                       
animal_id               Unique id for animal                                                                                  
animal_type             Cat, dog, or other (including at least one bat!).                                                         
breed                   Animal breed. Many animals are generic mixed-breeds, e.g. "Long-haired mix".  
color                   Color of the animal's fur, if it has fur.                                                                      
date_of_birth           date of birth of animal                                                                                         datetime                time of birth of animal                                                                                         monthyear               Month and year of birth of animal                                                                                
name                    Name of animal                                                                                                     outcome_subtype         Subtype of animal at the time it left the shelter                                                                 outcome_type            Ultimate outcome for this animal. Possible entries include transferred,[mercy] euthanized, adopted.               sex_upon_outcome        Sex of animal at the time it left the shelter                                                                      count                   No. of animal                                                                                                     sex                     sex of animal                                                                                                     

There are some other column as well.

Conclusion 
By leveraging machine learning models and other data transformation techniques, we were able to identify several important features that help predict a shelter cat's adoption or transfer outcome as well as fit a decently accurate prediction model that can be used for future cases. 

