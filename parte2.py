# Suponha uma funcao f(x) dada por:
# f(x) = c1 + exp(c2*x) + c3*x**c4
# onde c1, c2, c3 e c4 sao constantes.
# Desenvolva uma rotina numerica que, a partir de valores fornecidos para as constantes c1, c2,
# c3 e c4, permita o usuario:
# 1. Encontrar uma raiz num intervalo [a,b] escolhendo o Metodo da Bissecao ou o Metodo de
# Newton (ponto de partida igual (a+b)/2), ou;
# 2. Calcular o valor de sua integral definida num intervalo [a,b] optando pela quadratura de
# Gauss (Gauss-Legendre) ou quadratura polinomial e podendo escolher o numero de
# pontos de integracao a serem usados entre 2 e 10, ou;
# 3. Calcular a derivada num ponto x=a, a partir de um ∆x fornecido, podendo escolher entre
# os metodos de diferencas finitas passo a frente, passo atras e diferenca central, e/ou,
# 4. Estimar o valor da derivada num ponto x=a pela extrapolacao de Richard (com p=1) a
# partir de dois valores de ∆x fornecidos.

# INPUTS do Programa (arquivo de entrada):
# a) ICOD relativo a tarefa requerida (1- Raiz;2-Integral; 3-Derivada DF;4-Derivada RE )
# b) Constantes c1, c2, c3 e c4 mais os dados requeridos para cada tarefa;
# c) TOLm - tolerancia maxima para a solucao iterativa (para o item 1)
# OUTPUTS do Programa (arquivo de saida):
# a) Impressao dos dados lidos;
# b) Solucao obtida;
# c) Possiveis "erros de uso" (Possiblidade de nao convergencia, etc.)
# A entrega devera ser:
# 1. Um "pseudo" manual do usuario – orientacoes minimas de como usar o programa.

import argparse
from typing import Tuple
import numpy as np

def generate_f(c1: float, c2: float, c3: float, c4: float) -> callable:
    return lambda x: c1 + np.exp(c2*x) + c3*x**c4

def generate_f_derivative(c1: float,c2: float,c3: float,c4:float ) -> callable:
    return lambda x: c1*c2*np.exp(c2*x) + c3*c4*x**(c4-1)

def metodo_bissecao(f:callable, a: float, b: float, tol: float, NmaxIter: int) -> float: # a deve ser menor que b
    crescente = True
    print (f(a), f(b))
    if (f(a) < 0 and f(b) < 0) or (f(a) > 0 and f(b) > 0):
        print("Pode nao convergir")
    elif (f(a) < 0 and f(b) > 0):
       crescente = True
    elif (f(a) > 0 and f(b) < 0):
       crescente = False

    xi = 0.0
    while (b-a) > tol:
        if NmaxIter == 0:
            raise RuntimeError("Nao convergiu, numero maximo de iteracoes atingido")
        xi = (a+b)/2.0
        fi = f(xi)
        if (fi > 0.0):
            if crescente:
                b = xi
            else:
                a = xi
        else:
            if crescente:
                a = xi
            else:
                b = xi
        NmaxIter -= 1
    return xi

def metodo_newton(f:callable,f_derivative:callable, tol: float, a: float, b: float, NmaxIter: int) -> float:
    x0 = (a+b)/2.0
    for i in range(NmaxIter):
        x1 = x0 - f(x0)/f_derivative(x0)
        
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
    raise RuntimeError("Nao convergiu, numero maximo de iteracoes atingido")

def metodo_newton_secante(f:callable, a: float, b: float, tol: float, NmaxIter: int) -> float:
    x0 = (a+b)/2.0
    x1 = x0 + (a+b)/3.0
    fa = f(x0)
    for _ in range(NmaxIter):
        fi = f(x1)
        x1 = x0 - fi*(x1-x0)/(fi-fa)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1

    raise RuntimeError("Nao convergiu, numero maximo de iteracoes atingido")

def quadratura_polinomial(f:callable, a: float, b: float, number_of_points: int) -> float:
    vetor_simposon = []
    matriz_vandermonde = []
    resultado = 0.0

    deltax = float(abs(b-a)) / (number_of_points-1)
    for i in range(1,number_of_points + 1):
        vetor_simposon.append((b**i - a**i)/i)       
        vandemonde_line = []
        for j in range(number_of_points):
            xi = a+(j)*deltax
            vandemonde_line.append(xi**(i-1))
        matriz_vandermonde.append(vandemonde_line)

    matriz_vandermonde = np.array(matriz_vandermonde)
    vetor_simposon = np.array(vetor_simposon)
    vetor_coeficientes = np.linalg.solve(matriz_vandermonde,vetor_simposon)
    
    for i in range(vetor_coeficientes.shape[0]):
        resultado += f(matriz_vandermonde[1][i])*vetor_coeficientes[i]
    
    return resultado

tabela_gauss_legendre = [
    [#2
        [1.0000000000000000 ,	-0.5773502691896257],
        [1.0000000000000000 ,	0.5773502691896257],
    ],
    [#3
        [0.8888888888888888 ,	0.0000000000000000],
 	    [0.5555555555555556 ,	-0.7745966692414834],
 	    [0.5555555555555556 ,	0.7745966692414834],
    ],
    [#4
        [0.6521451548625461 ,	-0.3399810435848563],
        [0.6521451548625461 ,	0.3399810435848563],
        [0.3478548451374538 ,	-0.8611363115940526],
        [0.3478548451374538 ,	0.8611363115940526],
    ],
    [#5
        [0.5688888888888889 ,	0.0000000000000000],
        [0.4786286704993665 	,-0.5384693101056831],
        [0.4786286704993665 ,	0.5384693101056831],
        [0.2369268850561891 	,-0.9061798459386640],
        [0.2369268850561891 ,	0.9061798459386640],
        
    ],
    [#6
        [0.3607615730481386, 	0.6612093864662645],
        [0.3607615730481386, 	-0.6612093864662645],
        [0.4679139345726910, 	-0.2386191860831969],
        [0.4679139345726910, 	0.2386191860831969],
        [0.1713244923791704, 	-0.9324695142031521],
        [0.1713244923791704, 	0.9324695142031521],
    ],
    [#7
        [0.4179591836734694 ,	0.0000000000000000],
        [0.3818300505051189 ,	0.4058451513773972],
        [0.3818300505051189 ,	-0.4058451513773972],
        [0.2797053914892766 ,	-0.7415311855993945],
        [0.2797053914892766 ,	0.7415311855993945],
        [0.1294849661688697 ,	-0.9491079123427585],
        [0.1294849661688697 ,	0.9491079123427585] ,      
        
    ],
    [#8
        [0.3626837833783620, 	-0.1834346424956498],
        [0.3626837833783620, 	0.1834346424956498],
        [0.3137066458778873, 	-0.5255324099163290],
        [0.3137066458778873, 	0.5255324099163290],
        [0.2223810344533745, 	-0.7966664774136267],
        [0.2223810344533745, 	0.7966664774136267],
        [0.1012285362903763, 	-0.9602898564975363],
        [0.1012285362903763, 	0.9602898564975363],
    ],
    [#9
        [0.3302393550012598, 	0.0000000000000000],
        [0.1806481606948574, 	-0.8360311073266358],
        [0.1806481606948574, 	0.8360311073266358],
        [0.0812743883615744, 	-0.9681602395076261],
        [0.0812743883615744, 	0.9681602395076261],
        [0.3123470770400029, 	-0.3242534234038089],
        [0.3123470770400029, 	0.3242534234038089],
        [0.2606106964029354, 	-0.6133714327005904],
        [0.2606106964029354, 	0.6133714327005904],
    ],
    [#10
        [0.2955242247147529, 	-0.1488743389816312],
        [0.2955242247147529, 	0.1488743389816312],
        [0.2692667193099963, 	-0.4333953941292472],
        [0.2692667193099963, 	0.4333953941292472],
        [0.2190863625159820, 	-0.6794095682990244],
        [0.2190863625159820, 	0.6794095682990244],
        [0.1494513491505806, 	-0.8650633666889845],
        [0.1494513491505806, 	0.8650633666889845],
        [0.0666713443086881, 	-0.9739065285171717],
        [0.0666713443086881, 	0.9739065285171717],
    ],
    
]

def quadratura_gauss_legendre(f:callable, a: float, b: float, number_of_points: int) -> float:
    matriz_pesos = tabela_gauss_legendre[number_of_points - 2]
    resultado = 0.0
    L = (b - a)/2.0
    for i in range(len(matriz_pesos)):
        wi = matriz_pesos[i][0]
        xi = matriz_pesos[i][1]
        resultado += wi*f(L*xi + (b+a)/2.0)
    
    resultado *= L
    return resultado    

def diferencas_finitas_passo_afrente(f:callable, deltaX: float, x: float) -> float:
    return (f(x+deltaX) - f(x))/deltaX

def diferencas_finitas_passo_atras(f:callable, deltaX: float, x: float) -> float:
    return (f(x) - f(x-deltaX))/deltaX

def diferencas_finitas_passo_central(f:callable, deltaX: float, x: float) -> float:
    return (f(x+deltaX) - f(x-deltaX))/(2*deltaX)

def extrapolacao_richard(f:callable, deltaX1: float, deltaX2: float, x: float) -> float:
    q = deltaX1/deltaX2
    p = 1
    d1 = diferencas_finitas_passo_afrente(f, deltaX1, x)
    d2 = diferencas_finitas_passo_afrente(f, deltaX2, x)
    return d1 + ((d1 - d2)/((q**(-p))-1)) 

def calcula_raiz(f:callable,f_derivative:callable, a: float, b: float,deltax: float,x:float, tol: float, NmaxIter: int) -> float:
    if a is None:
        raise RuntimeError("a nao fornecido")
    if b is None:
        raise RuntimeError("b nao fornecido")
    
    input_metodo = int(input("Escolha o metodo de raiz:\n1- Bissecao;\n2- Newton;\n"))
    match input_metodo:
        case 1:
            return metodo_bissecao(f=f, a=a, b=b, tol=tol, NmaxIter=NmaxIter)
        case 2:
            return metodo_newton(f=f,f_derivative=f_derivative,a=a, b=b, tol=tol,NmaxIter=NmaxIter)
        case _:
            raise RuntimeError("Opcao invalida")

def calcula_integral(f:callable, a: float, b: float) -> float:
    if a is None:
        raise RuntimeError("a nao fornecido")
    if b is None:
        raise RuntimeError("b nao fornecido")
    # input_n_pontos = int(input("Escolha quantos pontos de integracao: "))
    # print("legendre: ",quadratura_gauss_legendre(f=f, a=a, b=b, number_of_points=input_n_pontos))
    # print("polinomial: ",quadratura_polinomial(f=f, a=a, b=b, number_of_points=input_n_pontos))
    
    input_n_pontos = int(input("Escolha quantos pontos de integracao: "))
    if input_n_pontos < 2 or input_n_pontos > 10:
        raise RuntimeError("Numero de pontos de integracao invalido")
    
    input_metodo = int(input("Escolha o metodo de integral:\n1- Gauss-Legendre;\n2- Quadratura Polinomial;\n"))
    match input_metodo:
        case 1:
            return quadratura_gauss_legendre(f=f, a=a, b=b, number_of_points=input_n_pontos)
        case 2:
            return quadratura_polinomial(f=f, a=a, b=b, number_of_points=input_n_pontos)
        case _:
            raise RuntimeError("Opcao invalida")

def calcula_derivada_DF(f:callable, deltaX: float, x: float) -> float:
    input_metodo = int(input("Escolha o metodo de diferencas finitas: \n 1. Passo frente \n 2. Passo atras \n 3. Passo central \n"))
    match input_metodo:
        case 1:
            return diferencas_finitas_passo_afrente(f, deltaX, x)
        case 2:
            return diferencas_finitas_passo_atras(f, deltaX, x)
        case 3:
            return diferencas_finitas_passo_central(f, deltaX, x)
        case _:
            raise RuntimeError("Opcao invalida")

def main():
    parser = argparse.ArgumentParser(description='Programa 2 de Algebra Linear')
    parser.add_argument('-c1', '--c1', type=float, help='c1',required=True)
    parser.add_argument('-c2', '--c2', type=float, help='c2',required=True)
    parser.add_argument('-c3', '--c3', type=float, help='c3',required=True)
    parser.add_argument('-c4', '--c4', type=float, help='c4',required=True)
    parser.add_argument('-a', '--a', type=float, help='a')
    parser.add_argument('-b', '--b', type=float, help='b')
    parser.add_argument('-ic', '--icod', type=int, help='ICOD relativo ao método de análise; 1- Raiz;2-Integral; 3-Derivada DF;4-Derivada RE',required=True)
    parser.add_argument('-Ax1', '--deltax1', type=float, help='DeltaX fornecido;')
    parser.add_argument('-Ax2', '--deltax2', type=float, help='DeltaX fornecido;')
    parser.add_argument('-it', '--tol', type=float, help='TOLm',default=0.0001)
    parser.add_argument('-mi', '--NmaxIter', type=int, help='Numero de iteracoes maximas',default=10000)
    parser.add_argument('-x', '--x', type=float, help='Ponto x')
    
    args = parser.parse_args()
    
    TOLm = args.tol
    ICOD = args.icod
    deltax1 = args.deltax1
    deltax2 = args.deltax2
    a = args.a
    b = args.b
    c1 = args.c1
    c2 = args.c2
    c3 = args.c3
    c4 = args.c4
    NmaxIter = args.NmaxIter
    x = args.x

    f = generate_f(c1=c1,c2=c2,c3=c3,c4=c4)
    #f = lambda x: 2 + x + 2*x**2 
    # f = lambda x: np.exp(-x**2)  
    match ICOD:
        case 1:
            f_derivative = generate_f_derivative(c1=c1,c2=c2,c3=c3,c4=c4)
            print("Raiz: ", calcula_raiz(f=f,f_derivative=f_derivative, a=a, b=b, tol=TOLm,NmaxIter=NmaxIter,x=x,deltax=deltax1))
        case 2:
            print("Integral: ", calcula_integral(f=f, a=a, b=b))
        case 3:
            if deltax1 is None:
                raise RuntimeError('deltax1 nao fornecido') 
            if x is None:
                raise RuntimeError('x nao fornecido')
            print("Derivada DF: ", calcula_derivada_DF(f=f, deltaX=deltax1, x=x))

        case 4:
            if deltax1 is None:
                raise RuntimeError('deltax1 nao fornecido')
            if deltax2 is None:
                raise RuntimeError('deltax2 nao fornecido')
            if not x:
                raise RuntimeError('x nao fornecido')
            print('Derivada RE: ',extrapolacao_richard(f=f, deltaX1=deltax1, deltaX2=deltax2, x=x))
            
        case _:
            raise RuntimeError('ICOD invalido')
if __name__ == '__main__':
    main()