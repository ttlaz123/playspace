import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import matplotlib.animation as animation
import scipy.sparse as sparse
import time

def generate_laplacian(system_size, hopping_strength=1):
    '''
    returns off diagonal 1s 
    '''
    H = np.zeros((system_size, system_size))
    np.fill_diagonal(H[1:], hopping_strength*np.ones(system_size-1))
    np.fill_diagonal(H[:,1:], hopping_strength*np.ones(system_size-1))

    return H

def generate_function(system_size, function='linear', params=None):
    midpoint = system_size/2
    if(function is None):
        y = np.zeros(system_size)
        funcstr = 'y=0'
    elif(function == 'linear'):
        y = [params['a']*(x-midpoint) for x in range((system_size))]
        funcstr = 'y=' + str(params['a']) +'*x'
    elif(function == 'quadratic'):
        y = [params['a']*(x-midpoint) + params['b']*np.power(x-midpoint, 2) 
                    for x in range((system_size))]
        funcstr = 'y=' + str(params['a']) +'x' +'+' +str(params['b']) +'x^2'
    elif(function == 'cubic'):
        y = [params['a']*(x-midpoint) + 
             params['b']*np.power(x-midpoint, 2) + 
            params['c']*np.power(x-midpoint, 3) 
                    for x in range((system_size))]
        funcstr = ('y=' + str(params['a']) +'x' +'+' +str(params['b']) +'x^2'+
                              + str(params['c']) + 'x^3')
    elif(function == 'quartic'):
        y = [params['a']*(x-midpoint) + 
             params['b']*np.power(x-midpoint, 2) + 
            params['c']*np.power(x-midpoint, 3) + 
             params['d']*np.power(x-midpoint, 4)
              for x in range((system_size))]
        funcstr = ('y=' + str(params['a']) +'x' +'+' +str(params['b']) +'x^2'+
                              + str(params['c']) + 'x^3' + + str(params['d']) + 'x^4')
    elif(function == 'sine'):
        y = [params['a']*np.sin(params['b']*(x-midpoint)) 
             for x in range((system_size))]
        funcstr = 'y=' + str(params['a']) +'sin(' +str(params['b']) +'x)'
    elif(function == 'cosine'):
        y = [params['a']*np.cos(params['b']*(x-midpoint)) 
             for x in range((system_size))]
        funcstr = 'y=' + str(params['a']) +'cos(' +str(params['b']) +'x)'
    elif(function == 'random'):
        y = 2*params['a']*np.random.rand(system_size)-params['a']
        funcstr = 'y=rand[-' + str(params['a']) +',' + str(params['a']) + ']'
    elif(function == 'step'):
       y = params['a']*np.heaviside(np.arange(system_size)-system_size/2, 0)-params['a']/2
       funcstr = 'y=-'+str(params['a']/2)+ '(x<0),'+str(params['a']/2)+ '(x>0)'
    H = np.diag(y)
    return H, funcstr, y

def generate_hamiltonian(system_size, function, params, hopping_strength=1):
    H1 = generate_laplacian(system_size, hopping_strength=hopping_strength)
    
    H0, funcstr, func = generate_function(system_size, function, params=params)
    return H1+H0, funcstr, func

def generate_hamiltonian_pair(system_size, function, params, hopping_strength=1):
    H1 = generate_laplacian(system_size, hopping_strength=hopping_strength)
    
    H0, funcstr, func = generate_function(system_size, function, params=params)
    funcstr2 = '-' + funcstr
    return func, H1+H0, H1-H0, funcstr, funcstr2

def time_evolve(H, psi0, ts):
    psis = []
    for t in ts:
        psit = np.matmul(expm(1j*t*H),psi0)
        psis.append(psit)
    return psis

def reconstruct_state(basis_state, evecs):
  try:
    size = (basis_state.shape[0])
    components = [basis_state[i,0]*(evecs[:,i]) for i in range(size)]
  except AttributeError:
    size = len(basis_state)
    components = [basis_state[i]*(evecs[:,i]) for i in range(size)]
  return np.round(np.sum(components, axis=0), decimals=10)

def time_evolve_fast(evecs, evals, state, ts):
  v1 = sparse.csr_matrix(evecs).transpose().conjugate()
  state = sparse.csr_matrix(state)
  try:
    basis_state = v1*state 
  except ValueError:
    basis_state = v1*state.transpose().conjugate()
  reconstructed = reconstruct_state(basis_state, evecs)
  evolved_states = []
  for t in ts:
    U = [np.exp(-1j*evals[i]*t) for i in range(len(evals))]
    assert(len(U) == basis_state.shape[0])
    evolved_states.append([U[i]*basis_state[i,0] for i in range(len(U))])
  time_evolved = []
  for s in evolved_states:
    reconstructed = reconstruct_state(s, evecs)
    time_evolved.append(reconstructed)
  return time_evolved 

def get_eigenvectors(H):
    return np.linalg.eig(H)

def conj_squared_fast(psi):
    return np.real(np.conj(psi)*psi)
def conj_squared(psi):
    return np.abs(psi)**2

def animate_evolution(psis1, psis2, ts, title, func, xlim=None, ylim=None):
    midpoint = len(psis1[0])/2
    fig, ax = plt.subplots()
    x = np.arange(len(psis1[0]))
    line1, = ax.plot(x-midpoint, conj_squared_fast(psis1[0]), linewidth=3, label='psi1')
    line2, = ax.plot(x-midpoint, conj_squared_fast(psis2[0]), label='psi2')
    text = ax.text(0.2, 0.8,  't=' + str(ts[0]), transform=ax.transAxes)
    
    pos_dict = {'max_pos': np.max(np.nonzero(psis1[0])) - midpoint,
                'min_pos' : np.min(np.nonzero(psis1[0])) - midpoint}
    
    max_pos = pos_dict['max_pos']
    min_pos = pos_dict['min_pos']
    pos_dict['min_posval'] = func[int(min_pos+midpoint)]
    pos_dict['max_posval'] = func[int(max_pos+midpoint)]
    max_line, = ax.plot([max_pos, max_pos], [0,1], linewidth=3, label='Max Pos')
    min_line, = ax.plot([min_pos, min_pos], [0,1], linewidth=3, label='Min Pos')
    text_min = ax.text(0.1, 0.7,  'minpos=' + str(min_pos), transform=ax.transAxes)
    text_minval = ax.text(0.1, 0.65,  'minval=' + str(pos_dict['min_posval']), transform=ax.transAxes)
    text_max = ax.text(0.1, 0.6,  'maxpos=' + str(max_pos), transform=ax.transAxes)
    text_maxval = ax.text(0.1, 0.55,  'maxval=' + str(pos_dict['max_posval']), transform=ax.transAxes)
    def animate(i):
        new_max_pos = np.max(np.nonzero(psis1[i])) - midpoint
        new_min_pos = np.min(np.nonzero(psis1[i])) - midpoint
        # for some reason dictionaries are necessary here
        if(new_max_pos > pos_dict['max_pos']):
            pos_dict['max_pos'] = new_max_pos
            pos_dict['max_posval'] = func[int(pos_dict['max_pos']+midpoint)]
        if(new_min_pos < pos_dict['min_pos']):
            pos_dict['min_pos'] = new_min_pos
            pos_dict['min_posval'] = func[int(pos_dict['min_pos']+midpoint)]
        max_line.set_xdata([pos_dict['max_pos'], pos_dict['max_pos']])
        min_line.set_xdata([pos_dict['min_pos'], pos_dict['min_pos']])
        line1.set_ydata(conj_squared_fast(psis1[i]))
        line2.set_ydata(conj_squared_fast(psis2[i]))
        text.set_text('t=' +str(ts[i]))
        text_min.set_text('minpos=' + str(pos_dict['min_pos']))
        text_max.set_text('maxpos=' + str(pos_dict['max_pos']))
        text_minval.set_text('minposval=' + str(pos_dict['min_posval']))
        text_maxval.set_text('maxposval=' + str(pos_dict['max_posval']))
        return line1, line2, max_line, min_line

    ani = animation.FuncAnimation(
        fig, animate, frames=len(psis1)#, interval=20, blit=True, save_count=50
    )
    print('Done Animating')
    if(xlim is not None):
       plt.xlim(xlim)
    if(ylim is not None):
       plt.ylim(ylim)
    plt.ylabel('|psi| ^2')
    plt.xlabel('Position (x)')
    plt.title(title)
    plt.legend()
    print('Saving animation:' + str(title))
    time1 = time.time()
    writergif = animation.PillowWriter(fps=10)
    ani.save(title + '.gif',writer=writergif)
    print('Saving done after time (s): ' + str(time1-time.time()))
    plt.show()

def run_localization_test(system_size, function, hopping_strength, psi0,  max_t, params1, params2,
                          xlim=None, ylim=None):
    midpoint = int(system_size/2)
    func, H, H2, funcstr, funcstr2 = generate_hamiltonian_pair(system_size, function, params1, 
                                      hopping_strength=hopping_strength)
    time1 = time.time()
    print('Finding eigenvectors of :')
    print(str(H))
    evals, evecs = get_eigenvectors(H)
    evals2, evecs2 = get_eigenvectors(H2)
    time2 = time.time()
    print('Done after time(s): ' + str(time2-time1))
    
    print('__________Time Evolving____________')
    time3 = time.time()
    ts = np.array(range(max_t))
    psis1 = time_evolve_fast(evecs, evals, psi0, ts)
    time4 = time.time()
    print('Done ' + funcstr + ' after time (s):' + str(time4-time3))
    psis2 = time_evolve_fast(evecs2, evals2, psi0, ts)
    time5 = time.time()
    print('Done ' + funcstr2 + ' after time (s):' + str(time5-time4))

    energy = np.conjugate(psis1[0]) @ H @ psis1[0]
    print('Energy:' + str(energy))
    title = funcstr + '_and_' + funcstr2+'_hoppingstrength='+str(hopping_strength)

    animate_evolution(psis1, psis2, ts, title, func, xlim, ylim)
                      
def main():
    system_size = 1000
    
    function = 'step'
    params1 = {'a': 50, 'b':0.001, 'c':1, 'd': 1}
    hopping_strength=1
    max_t = 600
    xlim = [-system_size/2, system_size/2]
    #xlim = [150, 250]
    ylim = [0, 1]
    params2 = {'a': -1, 'b':-0.1, 'c':1, 'd': 1}
    psi0 = np.zeros(system_size)
    psi0[700] = 1#/np.sqrt(2)
    #psi0[200] = 1/np.sqrt(2)
    run_localization_test(system_size, function, hopping_strength, psi0, max_t,
                          params1, params2,
                          xlim=xlim, ylim=ylim)
if __name__ == '__main__':
    main()

