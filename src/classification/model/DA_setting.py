from . import DA

def get_da_list(da):
    base_da_list = [[DA.NON_DA]]
    da_list = [[DA.NON_DA]]

    if (da & 0x0001) != 0:
        da_list.append([DA.H_FLIP])
    
    if (da & 0x0002) != 0:
        da_list.append([DA.V_FLIP])

    if (da & 0x0004) != 0:
        da_list.append([DA.ROTATE, 90])
        da_list.append([DA.ROTATE, 180])
        da_list.append([DA.ROTATE, 270])

    if (da & 0x0008) != 0:
        da_list.append([DA.ROTATE, 45])
        da_list.append([DA.ROTATE, 135])
        da_list.append([DA.ROTATE, 225])
        da_list.append([DA.ROTATE, 315])

    if (da & 0x0010) != 0:
        da_list.append([DA.ROTATE, 30])
        da_list.append([DA.ROTATE, 60])
        da_list.append([DA.ROTATE, 120])
        da_list.append([DA.ROTATE, 150])
        da_list.append([DA.ROTATE, 210])
        da_list.append([DA.ROTATE, 240])
        da_list.append([DA.ROTATE, 300])
        da_list.append([DA.ROTATE, 330])

    if (da & 0x0020) != 0:
        base_da_list.append([DA.GAUSSIAN_NOISE])
    
    if (da & 0x0040) != 0:
        base_da_list.append([DA.BLUR])

    if (da & 0x0080) != 0:
        base_da_list.append([DA.CONTRAST, 0])
        base_da_list.append([DA.CONTRAST, 1])

    return [base_da_list, da_list]

def make_info_number(augmantation_params):
    da = 0
    if augmantation_params["horizontal_flip"] == True:
        da += 0x0001
    if augmantation_params["vertical_flip"] == True:
        da += 0x0002
    if augmantation_params["rotate_90"] == True:
        da += 0x0004
    if augmantation_params["rotate_45"] == True:
        da += 0x0008
    if augmantation_params["rotate_30"] == True:
        da += 0x0010
    if augmantation_params["gaussian_noise"] == True:
        da += 0x0020
    if augmantation_params["blur"] == True:
        da += 0x0040
    if augmantation_params["contrast"] == True:
        da += 0x0080
    
    return da

def run(params):
    info_number = make_info_number(params)
    da_lists = get_da_list(info_number)
    return da_lists