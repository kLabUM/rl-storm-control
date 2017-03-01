import linecache

def line_idf(string, input_file):
    """Identify the Part of the input file to be changed"""
    """
    Arguments
        string     = Field to be modified
        input_file = input file to be modified
    --------------------------------------------------
    Outputs
        line       = Line number of the section
    """
    input_file = open(input_file, 'r')
    lines = 0
    for line in input_file:
        lines += 1
        if line[1:11] == string:
            line_number = (lines-1)
    input_file.close()
    return line_number


def idf_value(line_number, field_modify, input_file):
    """
    Arguments
        line_number  = Field Line Number
        input_file   = input file to be modified
        field_modify = Attribute to modify
    --------------------------------------------------
    Outputs
        Character    = Attribute value
    """
    line = linecache.getline(input_file, line_number+1)
    line = list(line)
    str_test = list(field_modify)
    for i in range(len(line)):
        if line[i] == str_test[0]:
            if line[i:i+len(str_test)] == str_test:
                print(i)
                break
    return i


def storm_event(input_file_1, output_file, field, series_name, time, intensity):
    """
    T
    """
    output_file = open(output_file, 'w')
    input_file = open(input_file_1, 'r')
    line_number = line_idf(field, input_file_1)
    i1 = 0
    while i1 <= line_number+3:
        line_1 = linecache.getline(input_file_1, i1)
        output_file.write(line_1)
        i1 = i1 + 1
    for i in range(len(time)):
        line = series_name + (16-len(series_name))*" " + 12*" " + time[i] + (11-len(time[i]))*" " + intensity[i] + (10-len(intensity[i]))*" " + "\n"
        output_file.write(line)

    output_file.write('\n')

    write = False
    for line in input_file:
        temp = line
        temp = list(temp)

        if temp[0] == '[' and temp[1] == "R" and temp[2] == "E":
            write = True
        if write:
            output_file.write(line)
    output_file.close()
    input_file.close()


if __name__ == "__main__":

    time = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    intensity = ["0", "5", "6", "7", "10", "12", "14", "12", "11", "10", "9", "8", "0"]
    storm_event('Parallel.inp', "Parallel.inp", 'TIMESERIES',
                'T1', time, intensity)
