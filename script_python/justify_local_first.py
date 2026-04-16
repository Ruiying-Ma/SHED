import os 


def justify_local_first(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()


    schema = lines[0].strip().split(',')

    tot_num = len(lines) - 1
    
    has_true_sht_idx = schema.index('has_vp')
    has_true_sht_num = sum(1 for line in lines[1:] if line.strip().split(',')[has_true_sht_idx] == 'True')

    wf_idx = schema.index('is_well_formatted')
    wf_num = sum(1 for line in lines[1:] if line.strip().split(',')[wf_idx] == 'True')

    lf_idx = schema.index('is_loosely_formatted')
    lf_num = sum(1 for line in lines[1:] if line.strip().split(',')[lf_idx] == 'True')

    da_idx = schema.index('is_depth_aligned')
    da_num = sum(1 for line in lines[1:] if line.strip().split(',')[da_idx] == 'True')

    local_idx = schema.index('is_local_first')
    local_num = sum(1 for line in lines[1:] if line.strip().split(',')[local_idx] == 'True')


    global_idx = schema.index('is_global_first')
    global_num = sum(1 for line in lines[1:] if line.strip().split(',')[global_idx] == 'True')

    print(f"Total number of samples: {tot_num}")
    print(f"Number of samples with true SHTs: {has_true_sht_num} ({has_true_sht_num / tot_num:.2%})")
    print(f"Number of well-formatted samples: {wf_num} ({wf_num / has_true_sht_num:.2%})")
    print(f"Number of loosely-formatted samples: {lf_num} ({lf_num / has_true_sht_num:.2%})")
    print(f"Number of depth-aligned samples: {da_num} ({da_num / has_true_sht_num:.2%})")
    print(f"Number of local-first samples: {local_num} ({local_num / has_true_sht_num:.2%})")
    print(f"Number of global-first samples: {global_num} ({global_num / has_true_sht_num:.2%})")


def local_first_adv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()


    schema = lines[0].strip().split(',')

    m_zip_stats = dict()
    for l in lines[1:]:
        fields = l.strip().split(',')
        zip_id = fields[schema.index('zip_id')]
        if zip_id not in m_zip_stats:
            m_zip_stats[zip_id] = {
                'total': 0,
                'has_true_sht_num': 0,
                'well_formatted_num': 0,
                'loosely_formatted_num': 0,
                'depth_aligned_num': 0,
                'local_first_num': 0,
                'global_first_num': 0
            }
        has_true_sht = fields[schema.index('has_vp')] == 'True'
        is_well_formatted = fields[schema.index('is_well_formatted')] == 'True'
        is_loosely_formatted = fields[schema.index('is_loosely_formatted')] == 'True'
        is_depth_aligned = fields[schema.index('is_depth_aligned')] == 'True'
        is_local_first = fields[schema.index('is_local_first')] == 'True'
        is_global_first = fields[schema.index('is_global_first')] == 'True'

        m_zip_stats[zip_id]['total'] += 1
        if has_true_sht:
            m_zip_stats[zip_id]['has_true_sht_num'] += 1
        if is_well_formatted:
            m_zip_stats[zip_id]['well_formatted_num'] += 1
        if is_loosely_formatted:
            m_zip_stats[zip_id]['loosely_formatted_num'] += 1
        if is_depth_aligned:
            m_zip_stats[zip_id]['depth_aligned_num'] += 1
        if is_local_first:
            m_zip_stats[zip_id]['local_first_num'] += 1
        if is_global_first:
            m_zip_stats[zip_id]['global_first_num'] += 1

    selected_zip_ids = []
    has_equal_num_ids = []
    for zip_id, stats in m_zip_stats.items():
        # print(f"Zip ID: {zip_id}")
        # print(f"  Total samples: {stats['total']}")
        if  stats['total'] < 1000:
            continue
            

        if stats['global_first_num'] >= stats['local_first_num']:
            continue
        else:
            selected_zip_ids.append(zip_id)


    print(f"Selected {len(selected_zip_ids)}: {selected_zip_ids}")
    tot_num = sum(m_zip_stats[zip_id]['total'] for zip_id in selected_zip_ids)
    has_true_sht_num = sum(m_zip_stats[zip_id]['has_true_sht_num'] for zip_id in selected_zip_ids)
    wf_num = sum(m_zip_stats[zip_id]['well_formatted_num'] for zip_id in selected_zip_ids)
    lf_num = sum(m_zip_stats[zip_id]['loosely_formatted_num'] for zip_id in selected_zip_ids)
    da_num = sum(m_zip_stats[zip_id]['depth_aligned_num'] for zip_id in selected_zip_ids)
    local_num = sum(m_zip_stats[zip_id]['local_first_num'] for zip_id in selected_zip_ids)
    global_num = sum(m_zip_stats[zip_id]['global_first_num'] for zip_id in selected_zip_ids)

    print(f"Total number of samples: {tot_num}")
    print(f"Number of samples with true SHTs: {has_true_sht_num} ({has_true_sht_num / tot_num:.2%})")
    print(f"Number of well-formatted samples: {wf_num} ({wf_num / has_true_sht_num:.2%})")
    print(f"Number of loosely-formatted samples: {lf_num} ({lf_num / has_true_sht_num:.2%})")
    print(f"Number of depth-aligned samples: {da_num} ({da_num / has_true_sht_num:.2%})")
    print(f"Number of local-first samples: {local_num} ({local_num / has_true_sht_num:.2%})")
    print(f"Number of global-first samples: {global_num} ({global_num / has_true_sht_num:.2%})")




if __name__ == "__main__":
    file_path = "/home/ruiying/SHTRAG/data/ccmain/ccmain_pdf_analysis.csv"
    # justify_local_first(file_path)
    local_first_adv(file_path)