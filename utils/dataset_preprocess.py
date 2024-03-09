def get_doc(datapoint, evidence_path, method="full"):
    docs = []
    if "EntityPages" in datapoint:
        entity_pages = datapoint["EntityPages"]
        for entity_page in entity_pages:
            if "Filename" in entity_page:
                filename = entity_page["Filename"]
                with open(f"{evidence_path}/wikipedia/{filename}", 'r') as myfile:
                    data = myfile.read()
                    docs.append(data)
    elif "SearchResults" in datapoint:
        search_results = datapoint["SearchResults"]
        for search_result in search_results:
            if "Filename" in search_result:
                filename = search_result["Filename"]
                with open(f"{evidence_path}/web/{filename}", 'r') as myfile:
                    data = myfile.read()
                    docs.append(data)
    else:
        print("No evidence found")
    return docs