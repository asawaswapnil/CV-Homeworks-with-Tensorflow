def parse_data(datadir):
    filenames_list=[]
    transcript_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.vtt'):
                filei = os.path.join(root, filename)
                filenames_list.append(filei)
                transcript_dict.append()
                ID_list.append(root.split('/')[-1])
