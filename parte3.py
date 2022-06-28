# Desenvolva uma rotina numerica para resolver a seguinte equacao diferencial pelo metodo
# Runge-Kutta-Nystron:

# INPUTS do Programa (num arquivo de entrada):
# a) Passo de integracao;
# b) Tempo total de integracao
# c) Valores dos parametros m,c e k e tambem de a1, a2, a3, w1, w2 e w3;
# Obs.: Desenvolva seus testes com m=1;c=0.1 e k=2; a1 = 1, a2 = 2, a3 = 1.5, w1 = 0.05,
# w2 = 1 e w3 = 2;
# OUTPUTS do Programa (num arquivo de saida):
# a) Impressao dos dados lidos;
# b) Solucao obtida (uma tabela com o tempo, deslocamento, velocidade e
# aceleracao);
# c) Caso possivel, seria interessante o usuario poder tambem visualizar os resultados
# anteriores;
# A entrega devera conter:
# 1. Um "pseudo" manual do usuario - orientacoes minimas de como usar o programa
import numpy as np
import argparse

def generate_f(a1, a2, a3, w1, w2, w3,c,k,m):
    f = lambda t: a1*np.sin(w1*t) + a2*np.sin(w2*t) + a3*np.cos(w3*t)
    return lambda t, x, dx: (f(t) - c*dx - k*x)/m

def f_exemplo(t,x,dx):
    return -9.8 - 1.0*dx*np.abs(dx)  

def range_kuta_nystron(f,dxk,xk,passo, tempo_total) -> list:
    
    tk = 0
    meio_passo = passo*0.5
    insantes_log = []
    while tk < tempo_total:
        result_funcao = f(tk,xk,dxk)
        
        insantes_log.append([tk,xk,dxk,result_funcao])
        
        K1 = meio_passo*result_funcao
        Q =  meio_passo*(dxk + (0.5*K1))
        K2 = meio_passo*f(tk+meio_passo, xk + Q, dxk + K1)
        K3 = meio_passo*f(tk+meio_passo, xk + Q, dxk + K2)
        L = passo * (dxk + K3)
        K4 = meio_passo*f(tk+passo, xk+L, dxk+2*K3)
        
        xk = xk + passo * (dxk + (1/3)*(K1 + K2 + K3))
        dxk = dxk + (1/3*(K1 + 2*K2 + 2*K3 + K4))
        tk = tk + passo
        
        # return tempo, deslocamento, velocidade e aceleracao
    result = np.array(insantes_log)
    return result

def main():
    parser = argparse.ArgumentParser(description='Programa 3 de Algebra Linear')
    parser.add_argument('-p', '--passo', type=float, help='Passo de integração;',required=True)
    parser.add_argument('-t', '--tempo', type=float, help='Tempo total de integração;',required=True)
    parser.add_argument('-m', '--m', type=float, help='Valor do parametro m;',required=True)
    parser.add_argument('-c', '--c', type=float, help='Valor do parametro c;',required=True)
    parser.add_argument('-k', '--k', type=float, help='Valor do parametro k;',required=True)
    parser.add_argument('-a1', '--a1', type=float, help='Valor do parametro a1;',required=True)
    parser.add_argument('-a2', '--a2', type=float, help='Valor do parametro a2;',required=True)
    parser.add_argument('-a3', '--a3', type=float, help='Valor do parametro a3;',required=True)
    parser.add_argument('-w1', '--w1', type=float, help='Valor do parametro w1;',required=True)
    parser.add_argument('-w2', '--w2', type=float, help='Valor do parametro w2;',required=True)
    parser.add_argument('-w3', '--w3', type=float, help='Valor do parametro w3;',required=True)
    args = parser.parse_args()

    m = float(args.m)
    c = float(args.c)
    k = float(args.k)
    a1 = float(args.a1)
    a2 = float(args.a2)
    a3 = float(args.a3)
    w1 = float(args.w1)
    w2 = float(args.w2)
    w3 = float(args.w3)
    passo = float(args.passo)
    tempo_total = float(args.tempo)

    f=generate_f(m=m,c=c, k=k, a1=a1, a2=a2, a3=a3, w1=w1, w2=w2, w3=w3)
    tabela = range_kuta_nystron(xk=0,dxk=0,tempo_total=tempo_total,passo=passo,f=f)
    print(tabela)
    np.savetxt('resultRKN.csv', tabela, delimiter=",", header="Tempo,Deslocamento,Velocidade,Aceleração", comments="" )   
    
if __name__ == '__main__':
    main()
# USAGE
# python3 parte3.py -p 0.01 -t 10 -m 1 -c 0.1 -k 2 -a1 1 -a2 2 -a3 1.5 -w1 0.05 -w2 1 -w3 
