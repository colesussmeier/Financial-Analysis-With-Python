# <center> State of The Economy 2019</center>

<center>Cole Sussmeier</center>

---

## Introduction 

I decided to do this project because I wanted to visualize the warning signs of a recession using data analytics. Since gold is often directly correlated economic and political uncertainty, I decided to make a simple algorithm to trade gold as well. I will assume that most people reading this will be more intersested in the results of this project rather than using the code, so I have attached two markdown files that walk through the projects. Please see markdown_files/GoldAnalysis.md first, as it justifies why the algorithm was written in the first place.

For anyone interested in creating an algorithm similar to this, I will have directions for installing the required software to do so at a later date. 

--- 

## To Do
* Implement more macroeconomic indicators
* More data for the price of gold-- Currently the algorithm is trading a gold ETF(SPDR Gold Shares) since it was the only free data I could find in the correct OHLCV format. This data only goes back to 2004
* optimization method-- Performance is not optimized, right now the weights of indicators are based on trial and error