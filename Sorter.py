"""DOCSTRING
Sigma Xi project
created 25 / 2 / 2025

adds a title / labels to a .CSV file
nothing more, nothing less

updated one for compatibility with the sk-learn model
"""

# imports
import io
import csv
import datetime
  
def convert_time_format(time_str):
  time = datetime.datetime.strptime(time_str, '%H:%M:%S.%f')
  time_24h = time.strftime('%H')
  
  return time_24h

def main() -> None:# public static void main(string[]args) {
  
  # file paths, put them here for easier readability but not neccessary
  DATASET_FILEPATH = r"/Users/71635/Documents/Computer Science/Sigma Xi/Ml model Meteor Showers/cleaned_meteor_data.csv"  # REPLACE THIS WITH YOUR DATASET FILE PATH
  OUTPUT_FILEPATH = r"/Users/71635/Documents/Computer Science/Sigma Xi/Ml model Meteor Showers/output_meteor_shower.csv" # REPLACE THIS WITH YOUR OUTPUT FILE PATH

  # loads the CSV's, in this case it open the meteor_data.csv and convert a string as a CSV
  labels = io.StringIO("time, range, height, vel, lat, long")
  dataset = open(DATASET_FILEPATH, 'r')

  # convert raw data into usable peices
  headers = csv.reader(labels)
  contents = csv.reader(dataset)

  count = 0
  
  # actual transferring of data
  with open(OUTPUT_FILEPATH, 'w', newline='') as output:
    writer = csv.writer(output)

    # write the headers / titles first
    for i in headers:
      writer.writerow(i)
    
    # then write the contents
    # ITEMS | time | range | height | vel | lat | long |
    # INDEX |  0   |   1   |   2    |  3  |  4  |  5   |  
    for i in contents:
      i = i[0].split() # convert from a singular string into a list of strings
      i = [i[j] for j in range(len(i)) if j not in [0, 2, 5, 6, 7, 11, 12]]
      
      i[0] = convert_time_format(i[0]) # convert time column
      i[1] = i[1].strip('0').strip('.') # removes leading and trailing spaces and zeroes
      i[2] = i[2].strip('0')
      i[3] = i[3].strip('0')
      i[5] = i[5].strip('0')
      
      writer.writerow(i)
      count += 1
      print(f" {count:5} | {i}")
      
      if count == 1216446:
        exit()

# this program is ment to be run, not imported
if __name__ == '__main__':
  main()