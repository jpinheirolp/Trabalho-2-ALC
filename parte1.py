# Desenvolva uma rotina computacional para solucionar numericamente o sistema de equacoees
# N.L. apresentado abaixo:

# A solucao significa encontrar as contantes c2, c3 e c4 para um dado conjunto de parametros
# θ1 e θ2 fornecidos.
# A rotina deve contemplar a possibilidade do usuario escolher entre utilizar o Metodo de
# Newton ou o Metodo de Broyden.


# INPUTS do Programa (arquivo de entrada):
# a) ICOD relativo ao metodo de analise (1- Newton;2-Broyden)
# b) Parametros θ1 e θ2
# c) TOLm - tolerância maxima para a solucao iterativa
# OUTPUTS do Programa (arquivo de saida):
# a) Impressao dos dados lidos;
# b) Solucao para as contantes c2, c3 e c4
# c) Possiveis "erros de uso" (Possiblidade de nao convergencia, etc.)

# Sugestão de valores iniciais (c2 = 1, c3 = 0 e c4 = 0).
# A entrega deverá conter um “pseudo” manual do usuário – orientações mínimas de como usar
# o programa e também a solução de três exemplos:

import argparse
from typing import Tuple
import numpy as np
import random

def generate_jacobian(c2 : float, c3:float, c4:float) -> np.matrix:

    J = np.matrix([
        [2*c2,4*c3,12*c4],
        [((12*c3*c2) + (36*c3*c4)), ((24*c3**2) + (6*c2**2) +(36*c2*c4) + (108*c4**2)) , (36*c3*c2) + (216*c3*c4)],
        [
            ( (120*(c3**2)*c2) + (576*c3**2*c4) + (454*(c4**2)*c2) + (1296*(c4**3)) + (72*(c2**2)*c4) + 3),  
            ( (240*c3**3) + (120*c3*c2**2) + (1152*c3*c2*c4) +  (4464*c3*c4**2) ),
            ( (576*c3**2*c2) + (4464*c3**2*c4) + (504*c4*c2**2) + (3888*c4**2*c2) + (13392*c4**3) + (24*c2**3) )
        ]
    ])
    return J

def generate_F_vector(c2 : float,c3:float, c4:float, teta1:float, teta2:float) -> np.ndarray:
    F = np.array(
        [c2**2+2*c3**2+6*c4**2 - 1.0,
        8*c3**3 + 6*c3*c2**2 + 36*c2*c3*c4 + 108*c3*c4**2 - teta1,
        (60*c3**4 + 60*c3**2*c2**2 + 576*c3**2*c4*c2 + 2232*c3**2*c4**2 + 252*c4**2*c2**2 + (1296*(c4**3)*c2) + (3348*(c4**4) + (24*(c2**3)*c4))+(3*c2)) - teta2
        ]
        )
    return F

def newthon_s_method_multidimensional(teta1:float,teta2:float, tol: float, NmaxIter: int) -> np.ndarray:
    x0 = np.array([1,0,0])
    res = np.inf
    xk = x0
    xk1 = x0
    
    while res > tol: 
        if NmaxIter == 0:
            raise RuntimeError("Nao convergiu, numero maximo de iteracoes atingido")
        xk = xk1[:]
        J = generate_jacobian(xk[0],xk[1],xk[2])
        F = generate_F_vector(xk[0],xk[1],xk[2],teta1,teta2)
        
        # sk =  metodo_lu(J,F)[1]
        sk = np.linalg.solve(J,-F)
        xk1 = xk + sk
        res = np.linalg.norm(sk,ord=2,axis=0) / np.linalg.norm(xk,ord=2,axis=0)
        # print(xk1,res)
        NmaxIter -= 1
    return xk1
    

def broyden_s_method_multidimensional(teta1:float,teta2:float, tol: float, NmaxIter: int) -> np.ndarray:
    res = np.inf
    xk = np.array([1,0,0])
    xk1 = np.array([1,0,0])
    Bk =  generate_jacobian(xk[0],xk[1],xk[2])
    Fk1 = generate_F_vector(xk[0],xk[1],xk[2],teta1,teta2)
    print(Bk,"\n","fk1",Fk1)

    while True: 
        if NmaxIter == 0:
            raise RuntimeError("Nao convergiu, numero maximo de iteracoes atingido")
        xk = xk1[:]
        Fk = Fk1
        # sk =  metodo_lu(J,F)[1]
        #sk = np.linalg.solve(Bk,-Fk)
        sk = -np.dot(np.linalg.inv(Bk) , (Fk))
        print(sk,"\n",Bk,"\n",Fk)
        xk1 = xk + sk
        Fk1 = generate_F_vector(xk1[0],xk1[1],xk1[2],teta1,teta2)
        yk = Fk1 - Fk
        res = np.linalg.norm(sk,ord=2,axis=0) / np.linalg.norm(xk,ord=2,axis=0)
        if res > tol:
            break
        print(np.matmul(Bk , sk),Bk , sk)
        Bk = Bk + np.dot((yk - np.matmul(Bk , sk)) , sk.T) * (1.0 / np.dot(sk.T , sk))
        print(Bk)
        # print(res)
        NmaxIter -= 1
        break
        
    return xk1


def main():
    parser = argparse.ArgumentParser(description='Programa 1 de Algebra Linear')
    parser.add_argument('-01', '--teta1', type=str, help='Teta 1 file',required=True)
    parser.add_argument('-02', '--teta2', type=str, help='Teta 2 file',required=True)
    parser.add_argument('-ic', '--icod', type=int, help='ICOD relativo ao metodo de análise; 1 - Newton; 2 - Broyden;',required=True)
    parser.add_argument('-it', '--tol', type=float, help='TOLm',default=0.0001)
    parser.add_argument('-mi', '--NmaxIter', type=int, help='Numero de iteracoes maximas',default=10000)

    args = parser.parse_args()
    
    TOLm = args.tol
    ICOD = args.icod
    NmaxIter = args.NmaxIter
    # parse file as np.array
    TETA_1 =  np.loadtxt(args.teta1, dtype=float, delimiter=' ')
    TETA_2 =  np.loadtxt(args.teta2, dtype=float, delimiter=' ')

    match ICOD:
        case 1:
            print('Newton')
            for i in range(TETA_1.shape[0]):
                print("input",i,":","teta1",TETA_1[i],"teta2",TETA_2[i])
            
            for i in range(TETA_1.shape[0]):
                result = newthon_s_method_multidimensional(teta1=TETA_1[i],teta2=TETA_2[i],tol=TOLm,NmaxIter=NmaxIter)
                print("output",i,":",result)
                # print("teste result in F vector:\n",generate_F_vector(result[0],result[1],result[2],TETA_1[i],TETA_2[i]))
            
        case 2:
            print('Broyden')
            for i in range(TETA_1.shape[0]):
                print("input",i,":","teta1",TETA_1[i],"teta2",TETA_2[i])
            
            for i in range(TETA_1.shape[0]):
                result = broyden_s_method_multidimensional(teta1=TETA_1[i],teta2=TETA_2[i],tol=TOLm,NmaxIter=NmaxIter)
                print("output",i,":",result)
            #for i in range(TETA_1.shape[0]):
            #    print("result for:",TETA_1[i],TETA_2[i])
            #    result = newthon_s_method_multidimensional(teta1=TETA_1[i],teta2=TETA_2[i],tol=TOLm, NmaxIter=NmaxIter)
            #    results.append(result)
                # print("teste result in F vector:\n",generate_F_vector(result[0],result[1],result[2],TETA_1[i],TETA_2[i]))
        case _:
            raise RuntimeError('ICOD invalido')
if __name__ == '__main__':
    main()