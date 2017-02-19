import csv


class Logger:
    def __init__(self, directory):
        self.my_file = open(directory + "log.csv", 'wb')
        self.wr = csv.writer(self.my_file, delimiter='|', quoting=csv.QUOTE_ALL)

    def write_log(self, current_time, human):
        self.wr.writerow([current_time, human.rect, human.found_speed])
        self.my_file.flush()

    def close(self):
        self.my_file.close()
