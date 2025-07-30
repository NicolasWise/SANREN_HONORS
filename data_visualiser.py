import csv
import matplotlib.pyplot as plt
import numpy

def read_CIS_data_csv(filename):
    data = []
    with open(filename, mode = 'r') as file:
        #csvfile = csv.reader(file)
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            name = row['Name']
            cis_value = row[' Core-Influence Strength Metric']
            data.append((name, cis_value))
            
    data.sort(key=lambda x:x[1])
    names, cis_values = zip(*data)
    return names, cis_values

def plot_line_graph(x, y, title=''):
    plt.figure(figsize=(12,6))
    plt.plot(x, y, marker='o', linestyle='-')

    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('CIS Value')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plots/Line_graph_{title}.png")
    

def plot_bar_graph(x, y, title=''):
    plt.figure(figsize=(12, 8))
    plt.barh(x, y, color = 'skyblue')
    plt.title(title)
    plt.xlabel('CIS Value')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Sample')
    plt.tight_layout()
    plt.savefig(f"Plots/Bar_graph_{title}.png")
            

def main():
    filename = 'Analyses/CIS_metric.csv'
    names, cis_values = read_CIS_data_csv(filename)
    plot_line_graph(names, cis_values, title='Core-Influence Strength metric per Sample')
    plot_bar_graph(names, cis_values, title='Core-Influence Strength metric per Sample')
    
if __name__ == '__main__':
    main()