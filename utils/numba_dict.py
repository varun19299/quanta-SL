import numba as nb


def python_to_numba_dict(pyObj):
    if type(pyObj) == dict:
        pyDict = pyObj
        keys = list(pyDict.keys())
        values = list(pyDict.values())

        # Keys
        if type(keys[0]) == str:
            nbhKeytype = nb.types.string
        elif type(keys[0]) == int:
            nbhKeytype = nb.types.int64
        elif type(keys[0]) == float:
            nbhKeytype = nb.types.float64
        else:
            raise ValueError(f"Key type {type(keys[0])} not considered.")

        # Values
        if type(values[0]) == int:
            nbh = nb.typed.Dict.empty(nbhKeytype, nb.types.int64)
            for i, key in enumerate(keys):
                nbh[key] = values[i]
            return nbh
        elif type(values[0]) == str:
            nbh = nb.typed.Dict.empty(nbhKeytype, nb.types.string)
            for i, key in enumerate(keys):
                nbh[key] = values[i]
            return nbh
        elif type(values[0]) == float:
            nbh = nb.typed.Dict.empty(nbhKeytype, nb.types.float64)
            for i, key in enumerate(keys):
                nbh[key] = values[i]
            return nbh
        elif type(values[0]) == dict:
            for i, subDict in enumerate(values):
                subDict = python_to_numba_dict(subDict)
                if i == 0:
                    nbh = nb.typed.Dict.empty(nbhKeytype, nb.typeof(subDict))
                nbh[keys[i]] = subDict
            return nbh
        elif type(values[0]) == list:
            for i, subList in enumerate(values):
                subList = python_to_numba_dict(subList)
                if i == 0:
                    nbh = nb.typed.Dict.empty(nbhKeytype, nb.typeof(subList))
                nbh[keys[i]] = subList
            return nbh
    elif type(pyObj) == list:
        pyList = pyObj
        data = pyList[0]
        if type(data) == int:
            nbs = nb.typed.List.empty_list(nb.types.int64)
            for data_ in pyList:
                nbs.append(data_)
            return nbs
        elif type(data) == str:
            nbs = nb.typed.List.empty_list(nb.types.string)
            for data_ in pyList:
                nbs.append(data_)
            return nbs
        elif type(data) == float:
            nbs = nb.typed.List.empty_list(nb.types.float64)
            for data_ in pyList:
                nbs.append(data_)
            return nbs
        elif type(data) == dict:
            for i, subDict in enumerate(pyList):
                subDict = python_to_numba_dict(subDict)
                if i == 0:
                    nbs = nb.typed.List.empty_list(nb.typeof(subDict))
                nbs.append(subDict)
            return nbs
        elif type(data) == list:
            for i, subList in enumerate(pyList):
                subList = python_to_numba_dict(subList)
                if i == 0:
                    nbs = nb.typed.List.empty_list(nb.typeof(subList))
                nbs.append(subList)
            return nbs
