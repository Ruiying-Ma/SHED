import os
import shutil

for folder_path in [
    "/home/ruiying/SHTRAG/data/finance/grobid-old/grobid",
    "/home/ruiying/SHTRAG/data/finance/grobid-old/node_clustering",
    "/home/ruiying/SHTRAG/data/finance/grobid-old/sbert.gpt-4o-mini.c100.s100/sht",
    "/home/ruiying/SHTRAG/data/finance/grobid-old/sbert.gpt-4o-mini.c100.s100/sht_vis",
]:
    new_folder_path = folder_path.replace("grobid-old", "grobid")
    os.makedirs(new_folder_path, exist_ok=True)

    for filename in sorted(os.listdir(folder_path)):
        new_filename = filename.replace(".grobid", "")
        new_path = os.path.join(new_folder_path, new_filename)
        assert not os.path.exists(new_path)
        assert os.path.exists(os.path.join(folder_path, filename))
        shutil.copy(os.path.join(folder_path, filename), new_path)