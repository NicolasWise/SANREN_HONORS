import csv
from scipy.stats import spearmanr

def read_nodes_from_csv(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Skip header
        return [row[0] for row in reader]
    
def write_test_to_csv(output, filename, first):
    mode = 'w' if first else 'a'
    with open(filename, mode, encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([output])

def sample_statistical_analysis():
    metrics = ['betweeness_centrality','closeness_centrality','degree_centrality', 'core_influences', 'core_numbers', 'core_strengths']
    samples = ['sample1','sample2','sample3','sample4','sample5','sample6','sample7','sample8']
    first_write = True

    for sample in samples:
        datasets = {}
        for metric in metrics:
            name = f'{metric}_{sample}'
            datasets[name] = read_nodes_from_csv(f'Analyses/top_r_{name}.csv')

        
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                name_i = f'{metrics[i]}_{sample}'
                name_j = f'{metrics[j]}_{sample}'
                list_i = datasets[name_i]
                list_j = datasets[name_j]

                '''Intersect and rank:
                    - This ensures that the spearman rank correlation test
                    only test the ranks of the common nodes between each dataset
                    due to the partial overlap in nodes between the datasets.'''
                common_nodes = list(set(list_i).intersection(list_j))
                if not common_nodes:
                    print(f"No overlap between {name_i} and {name_j}")
                    continue
                '''Re-rank common nodes between datasets.'''
                ranks_i = [list_i.index(node) for node in common_nodes]
                ranks_j = [list_j.index(node) for node in common_nodes]

                '''Apply Spearman rank correlation test'''
                coef, p = spearmanr(ranks_i, ranks_j)
                output = f"Spearman correlation between {name_i} and {name_j}: {coef:.3f}, p = {p:.3g}"
                write_test_to_csv(output, filename='Analyses/Spearman_rank_test_samples.csv', first=first_write)
                first_write = False

def main_statistical_analysis():
    names = ['betweeness_centrality','closeness_centrality','degree_centrality', 'core_influences', 'core_numbers', 'core_strengths']
    datasets = {}

    for name in names:
        datasets[name] = read_nodes_from_csv(f'Analyses/top_r_{name}.csv')

    first_write = True
    
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name_i = names[i]
            name_j = names[j]
            list_i = datasets[name_i]
            list_j = datasets[name_j]

            '''Intersect and rank:
                - This ensures that the spearman rank correlation test
                only test the ranks of the common nodes between each dataset
                due to the partial overlap in nodes between the datasets.'''
            common_nodes = list(set(list_i).intersection(list_j))
            if not common_nodes:
                print(f"No overlap between {name_i} and {name_j}")
                continue
            '''Re-rank common nodes between datasets.'''
            ranks_i = [list_i.index(node) for node in common_nodes]
            ranks_j = [list_j.index(node) for node in common_nodes]

            '''Apply Spearman rank correlation test'''
            coef, p = spearmanr(ranks_i, ranks_j)
            output = f"Spearman correlation between {name_i} and {name_j}: {coef:.3f}, p = {p:.3g}"
            write_test_to_csv(output, filename='Analyses/Spearman_rank_test.csv', first=first_write)
            first_write = False

def main():
    #main_names = 
    #main_statistical_analysis()
    sample_statistical_analysis()
            

if __name__ == '__main__':
    main()