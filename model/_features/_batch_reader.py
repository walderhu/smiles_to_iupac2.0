
def batch_reader(filepath, batch_size=5e4):
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n')
        batch = []
        for line in f:
            batch.append(line.rstrip('\n'))
            if len(batch) == batch_size:
                yield '\n'.join([header] + batch)
                batch = []
        if batch:
            yield '\n'.join([header] + batch)