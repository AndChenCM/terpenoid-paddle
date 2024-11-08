from threading import Thread, Lock
from rdkit.Chem import AllChem
from rdkit import Chem
from .compound_tools import mol_to_geognn_graph_data_raw3d, mol_to_geognn_graph_data_MMFF3d

mutex = Lock()

def transfer_smiles_to_graph_(smiles):
    n = len(smiles)
    global p

    while True:
        mutex.acquire()
        if p >= n:
            mutex.release()
            break
        index = p
        p += 1
        mutex.release()

        smi = smiles[index]
        mutex.acquire()
        if index % 100 == 0:
            print(index, ':', round(index/n*100, 2), '%', smi)
        mutex.release()
        try:
            mol = AllChem.MolFromSmiles(smi)
            mol_graph = mol_to_geognn_graph_data_MMFF3d(mol)
        except:
            print("Invalid smiles!", smi)
            continue

        global smiles_to_graphs
        mutex.acquire()
        smiles_to_graphs[smi] = mol_graph
        mutex.release()


def transfer_smiles_to_graph(smiles, num_worker):
    global smiles_to_graphs
    smiles_to_graphs = {}
    global p
    p = 0
    thread_count = num_worker
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=transfer_smiles_to_graph_, args=(smiles,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return smiles_to_graphs


def transfer_mol_to_graph_(mols, smiles):
    n = len(mols)
    global p

    while True:
        mutex.acquire()
        if p >= n:
            mutex.release()
            break
        index = p
        p += 1
        mutex.release()

        mol = mols[index]
        smi = smiles[index]
        mutex.acquire()
        if index % 100 == 0:
            print(index, ':', round(index/n*100, 2), '%', smi)
        mutex.release()
        try:
            mol_graph = mol_to_geognn_graph_data_raw3d(mol)
        except:
            print("Invalid smiles!", smi)
            continue

        global smiles_to_graphs
        mutex.acquire()
        smiles_to_graphs[smi] = mol_graph
        mutex.release()


def transfer_mol_to_graph(mols, smiles, num_worker):
    global smiles_to_graphs
    smiles_to_graphs = {}
    global p
    p = 0
    thread_count = num_worker
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=transfer_mol_to_graph_, args=(mols, smiles,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return smiles_to_graphs


