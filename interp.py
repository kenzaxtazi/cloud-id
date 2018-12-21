def intsbetween(q0, q1):
    out = []
    row0, col0 = q0
    row1, col1 = q1

    if row0 - row1 == 0:
        row = row0
        for col in range(min(col0, col1) + 1, max(col0, col1)):
            out.append([row, col])
    elif col0 - col1 == 0:
        col = col0
        for row in range(min(row0, row1) + 1, max(row0, row1)):
            out.append([row, col])
    else:
        for row in range(min(row0, row1), max(row0, row1) + 1):
            for col in range(min(col0, col1), max(col0, col1) + 1):
                truecol = col0 + ((col1 - col0)/(row1 - row0)) * (row - row0)
                if abs(truecol - col) <= 0.5:
                    out.append([row, col])
    return(out[1:-1])

if __name__ == '__main__': 
    q0 = [350, 2192]
    q1 = [352, 2994]
    print(intsbetween(q0, q1))