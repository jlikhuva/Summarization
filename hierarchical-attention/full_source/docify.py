with open('valid.doc.txt', 'w') as file:
    file.write('\n'.join(str(n) for n in range(0, 189652)))
    file.close()
with open('train.doc.txt', 'w') as file:
    file.write('\n'.join(str(n) for n in range(0, 3803958)))
    file.close()
