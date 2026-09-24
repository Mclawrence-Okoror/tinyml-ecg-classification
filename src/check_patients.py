import wfdb

records = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119","121",
    "122","123","124","200","201","202","203","205","207","208",
    "209","210","212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234"
]

print("=" * 60)
print("MIT-BIH RECORD INFORMATION")
print("=" * 60)

for record_name in records:
    header = wfdb.rdheader(
        "mit-bih-arrhythmia-database-1.0.0/" + record_name
    )

    print()
    print("RECORD:", record_name)

    for comment in header.comments:
        print(" ", comment)