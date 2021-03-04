#! /usr/bin/env python3
'''
Code & data associated with "Full propagation of analytical uncertainties in Δ47 measurements"
(M. Daëron, submitted to Geochemistry, Geophysics, Geosystems)
'''

__author__    = 'Mathieu Daëron'
__contact__   = 'daeron@lsce.ipsl.fr'
__copyright__ = 'Copyright (c) 2020 Mathieu Daëron'
__license__   = 'Modified BSD License - https://opensource.org/licenses/BSD-3-Clause'
__date__      = '2021-03-04'
__version__   = '1.1.0'


N_MONTECARLO = 10_000
SEED = 12345678

from pylab import *
from scipy import linalg, interpolate
from scipy.special import erf
from scipy.stats import chi2, norm, kstest
from matplotlib.patches import Ellipse
import matplotlib.patheffects as PathEffects
from numpy import random as nprandom
from statistics import stdev
from tqdm import tqdm

from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = "Helvetica"
rcParams["font.size"] = 10
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "sans"
rcParams["mathtext.bf"] = "sans:bold"
rcParams["mathtext.it"] = "sans:italic"
rcParams["mathtext.cal"] = "sans:italic"
rcParams["mathtext.default"] = "rm"
rcParams["xtick.major.size"] = 4
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 4
rcParams["ytick.major.width"] = 1
rcParams["axes.grid"] = False
rcParams["axes.linewidth"] = 1
rcParams["grid.linewidth"] = 1
rcParams["grid.linestyle"] = "-"
rcParams["grid.alpha"] = .15
rcParams["savefig.dpi"] = 150


def figure_margins(fig, left = 10, right = 10, bottom = 10, top = 10, *args, **kwargs):
	'''
	Sets subplots_adjust() to margins specified in mm.
	'''
	w,h = [x*25.4 for x in fig.get_size_inches()]
	x1 = left / w
	x2 = 1 - right / w
	y1 = bottom / h
	y2 = 1 - top / h
	return subplots_adjust(x1, y1, x2, y2, *args, **kwargs)


def read_csv( filename, VERBOSE = False ) :
	'''
	Reads the contents of a csv file and returns a list of dictionaries.
	'''
	with open( filename ) as f :
		data = [ [ smart_type( ll.strip() ) for ll in l.split(',') ] for l in f.readlines() ]
	data = [ { k:v for k,v in zip( data[0], d ) if v != '' } for d in data[1:] ]
	if VERBOSE :
		print(f'\n--- read_csv ---\n\t{len(data)} records read from {filename}.')
	return data

def smart_type( x ) :
	'''
	Tries to guess whether a variable is a float, an integer, or neither.
	'''    
	y = x
	if '.' in x :
		try : y = float(x)
		except : pass
	else :
		try : y = int(x)
		except : pass
	return y


def normalize(data, Nominal_D47, verbal = False):
	A = [
		[ Nominal_D47[r['Sample']], r['d47'], 1 ]
		for r in data if r['Sample'] in Nominal_D47
		]
	Y = [
		[ r['D47raw'] ]
		for r in data if r['Sample'] in Nominal_D47
		]
	A, Y = array(A), array(Y)
	
	CM = inv( A.T @ A )
	a,b,c = ( CM @ A.T @ Y ).T[0,:]

	for r in data :
		r['D47'] = ( r['D47raw'] - b * r['d47'] - c ) / a

	samples = sorted({r['Sample'] for r in data})
	Ns = len(samples)
	Na = len(data)
	avgD47 = {
		sample: mean([r['D47'] for r in data if r['Sample'] == sample])
		for sample in samples
		}
	chi2 = sum([(r['D47'] - avgD47[r['Sample']])**2 for r in data])
	CM *= a**2 * chi2 / (Na-Ns)
	rD47 = sqrt(chi2/(Na-Ns))
	
	if verbal:
# 		print(f'a = {a:.2f} ± {CM[0,0]**.5:.4f}')
# 		print(f'b = {b:.1e} ± {CM[1,1]**.5:.4e}')
# 		print(f'c = {c:.3f} ± {CM[2,2]**.5:.4f}')
# 		print(f'rD47 = {rD47:.4f} ‰')
# 		print(f'N = {Y.size}')
		print(f'rD47/sqrt(N) = {rD47 / (Y.size)**.5:.6f} ‰')

	return a, b, c, CM, rD47


def cov_ellipse( cov, q = .95 ):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float
        Confidence level, should be in (0, 1)

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    r2 = chi2.ppf(q, 2)
    val, vec = linalg.eigh(cov)
    width, height = 2 * sqrt(val[:, None] * r2)
    rotation = degrees(arctan2(*vec[::-1, 0]))

    return width, height, rotation


def covariance_of_unknowns(data):
	'''
	within a single session
	'''
	a, b, c, CM, rD47 = normalize(data, Nominal_D47)
	unknowns = sorted({r['Sample'] for r in data if r['Sample'] not in Nominal_D47})
	X = []
	autogenic = []
	Nu = len(unknowns)

	J = zeros((Nu, Nu+3))
	C = zeros((Nu+3, Nu+3))
	C[-3:,-3:] = CM[:,:]
	Cs = zeros((Nu+3, Nu+3))
	Cs[-3:,-3:] = CM[:,:]

	for k,u in enumerate(unknowns):
		U = [r for r in data if r['Sample'] == u]
		d47 = mean([r['d47'] for r in U])
		D47 = mean([r['D47'] for r in U])
		X += [D47]
		autogenic += [rD47 / len(U)**.5]
		C[k,k] = rD47**2 / len(U)
		J[k,k] = 1 / a
		J[k,-3] = -D47 / a
		J[k,-2] = -d47 / a
		J[k,-1] = -1 / a

	V = J @ C @ J.T
	Vs = J @ Cs @ J.T

	fig = figure(figsize=(5,5))
	figure_margins(fig, 17, 3, 15, 5, hspace = .1, wspace = .1)
	k = 0
	for i in range(Nu-1):
		for j in range(i+1):
			k += 1
			ax = subplot(3,3,(Nu-j-2)*3+Nu-i-1)

			CM = V[[i+1,j],:][:,[i+1,j]]		
			w,h,r = cov_ellipse( CM )
			ax.add_artist(
				Ellipse(
					xy = (X[i+1],X[j]), width = w, height = h, angle = r,
					lw = 1.5, fc = 'none', ec = 'r', ls = '-' ))

			CM = Vs[[i+1,j],:][:,[i+1,j]]
			w,h,r = cov_ellipse( CM )
			ax.add_artist(
				Ellipse(
					xy = (X[i+1],X[j]), width = w, height = h, angle = r,
					lw = .75, fc = 'none', ec = 'r', ls = '-' ))

			ax.add_artist(
				Ellipse(
					xy = (X[i+1],X[j]), width = 1.96*2*autogenic[i+1], height = 1.96*2*autogenic[j], angle = 0,
					lw = .75, fc = 'none', ec = 'r', ls = '--' ))

			plot(X[i+1], X[j], 'r+', mew=.75, ms = 4)
			
			e = 0.037
			axis([X[i+1] - e, X[i+1] + e, X[j] - e, X[j] + e])
			ax.xaxis.set_major_locator(MultipleLocator(0.03))
			ax.yaxis.set_major_locator(MultipleLocator(0.03))
			if not j:
				xlabel(f'{unknowns[i+1]}', weight = 'bold')
			else:
				ax.set_xticklabels(['' for x in xticks()[0]])

			if i == 2:
				ylabel(f'{unknowns[j]}', weight = 'bold')
			else:
				ax.set_yticklabels(['' for x in yticks()[0]])

	savefig('Fig3_covariance_of_unknowns.pdf')
	close(fig)

	S = diag(V).reshape((1,Nu))**.5
	V = V/S
	V = V/S.T
	
# 	tex = '\\begin{tabular}{'+'c'*Nu+'}\n\\toprule\n'
# 	tex += '& \\textbf{' + '} & \\textbf{'.join(unknowns[:-1]) + '} \\\\\n'
# 	for i in range(Nu-1):
# 		tex += f'\\textbf{{{unknowns[i+1]}}}'
# 		for j in range(i+1):
# 			tex += f' & ${V[i+1,j]:+.2f}$'
# 		tex += '\\\\\n'
# 	tex += '\\bottomrule\n\\end{tabular}'
# 	with open('covariance_of_samples.tex', 'w') as fid:
# 		fid.write(tex)

	return unknowns, V

def total_error(a, b, c, CM, d47, D47raw, sD47raw):
	D47 = (D47raw - b * d47 - c)/a
	J = array([1, -D47, -d47, -1]) / a
	C = zeros((4,4))
	C[0,0] = sD47raw**2
	C[1:,1:] = CM[:3,:3]
	sD47 = (J @ C @ J.T)**.5
	return float(sD47)

# def total_error_new(u, v, w, CM, d47, D47raw, sD47raw):
# 	D47 = u * D47raw + v * d47 + w
# 	J = array([u, D47raw, d47, 1])
# 	C = zeros((4,4))
# 	C[0,0] = sD47raw**2
# 	C[1:,1:] = CM[:,:]
# 	sD47 = (J @ C @ J.T)**.5
# 	return float(sD47)

def errormap(data, a, b, c, CM, Nominal_D47,
	manual = 'None',
	anchornumbers = False,
	addwidth = 0,
	cskip = 2,
	cstart = 1,
	verbal = False,
	):
	
	X = [r['d47'] for r in data if r['Sample'] in Nominal_D47]
	Y = [r['D47'] for r in data if r['Sample'] in Nominal_D47]
	kw = dict(marker = 'x', ls = 'None', ms = 4, mew = 2/3, mec = 'r', alpha = 1)
	plot(X, Y, **kw)
	X = [r['d47'] for r in data if r['Sample'] not in Nominal_D47]
	Y = [r['D47'] for r in data if r['Sample'] not in Nominal_D47]
	kw['mec'] = 'k'
	plot(X, Y, **kw)
	x1, x2, y1, y2 = axis()
	x1 -= addwidth
	x2 += addwidth

	n = 50 # change to higher spatial resolution (n = 500) to test whether rD47/sqrt(N) == SI.min()
	XI,YI = meshgrid(linspace(x1, x2, n), linspace(y1, y2, n))
	SI = array([[
		total_error(a, b, c, CM, xi, a*yi + b*xi + c, 0)
		for xi in XI[0,:]] for yi in YI[:,0]])
	cinterval = 1e-3
	cval = [k * cinterval for k in range(int(SI.max() // cinterval) + 1)][cstart::cskip]
	cs = contour( XI, YI, SI, cval, colors = 'r', alpha = .33, zorder = -100 )
	labels = clabel(cs, **({} if manual == 'None' else {'manual':manual}))
# 	plot(*zip(*manual), 'k+')
	for label in labels:
		label.set_zorder(-100)
	
	if verbal:
		print(f'min(SI) = {SI.min():.6f} ‰')

	axis([x1, x2, y1, y2])
	xlabel('δ$_{47}$ (‰)')
	ylabel('Δ$_{47}$ (‰)')

	if anchornumbers:
		t = text(0.03, 0.045,
			'\n'.join(
				[f'{anchor} (×{len([r for r in data if r["Sample"] == anchor])})'
				for anchor in Nominal_D47]),
			size = 8,
			weight = 'bold',
			va = 'bottom',
			transform=gca().transAxes,
			)
		t.set_bbox(dict(facecolor='w', alpha=1, edgecolor='k', lw=0.5, zorder = 200))

	xticks([-40, -20, 0, 20])
	yticks([.2, .4, .6])

def fig_properties_of_standardization_errors(session = 'Session02', lab = 'Lab12'):
	data = read_csv(f'rawdata.csv')
	data = [r for r in data if r['Lab'] == lab and r['Session'] == session]

	covariance_of_unknowns(data)
	
	a, b, c, CM, rD47 = normalize(data, Nominal_D47)

	fig = figure(figsize = (7,5))
	figure_margins(fig, 15, 5, 13, 8, hspace = .2, wspace = .1)

	ax = subplot(221)
	errormap(data, a, b, c, CM, Nominal_D47, [(2, 0.37), (-6, 0.51), (-15, 0.56), (-25, 0.56), (-36, 0.59)], anchornumbers = True)
	title(f'Original data ({lab}, {session})', size = 9, weight = 'bold')
# 	ax.set_xticklabels(['']*len(ax.get_xticklabels()))
	xlabel('')
	xticks([])


	N1 = len([r for r in data if r['Sample'] == 'ETH-1'])
	N2 = len([r for r in data if r['Sample'] == 'ETH-2'])
	N3 = len([r for r in data if r['Sample'] == 'ETH-3'])
	n = N3 // 2
	N1 -= (N3-n)
	N2 -= n
	data2 = (
		[r.copy() for r in data if r['Sample'] not in Nominal_D47]
		+ [r.copy() for r in data if r['Sample'] == 'ETH-1'][:N1]
		+ [r.copy() for r in data if r['Sample'] == 'ETH-2'][:N2]
		+ [r.copy() for r in data if r['Sample'] == 'ETH-3']
		+ [r.copy() for r in data if r['Sample'] == 'ETH-3']
		)
	a, b, c, CM, rD47 = normalize(data2, Nominal_D47)
	
	ax = subplot(222)
	errormap(data2, a, b, c, CM, Nominal_D47, [(8, 0.51), (-2, 0.52), (-10, 0.53), (-18, 0.54), (-25, 0.55), (-33, 0.56)], anchornumbers = True)
	title(f'Doubling the number of ETH-3 analyses', size = 9, weight = 'bold')
# 	ax.set_xticklabels(['']*len(ax.get_xticklabels()))
	xlabel('')
	xticks([])
# 	ax.set_yticklabels(['']*len(ax.get_yticklabels()))
	ylabel('')
	yticks([])
	
	N1 = len([r for r in data if r['Sample'] == 'ETH-1'])
	N2 = len([r for r in data if r['Sample'] == 'ETH-2'])
	N3 = len([r for r in data if r['Sample'] == 'ETH-3'])
	n = N1 // 2
	N2 -= (N1-n)
	N3 -= n
	data2 = (
		[r.copy() for r in data if r['Sample'] not in Nominal_D47]
		+ [r.copy() for r in data if r['Sample'] == 'ETH-1']
		+ [r.copy() for r in data if r['Sample'] == 'ETH-1']
		+ [r.copy() for r in data if r['Sample'] == 'ETH-2'][:N2]
		+ [r.copy() for r in data if r['Sample'] == 'ETH-3'][:N3]
		)
	a, b, c, CM, rD47 = normalize(data2, Nominal_D47)
	
	ax = subplot(223)
	errormap(data2, a, b, c, CM, Nominal_D47, [(7, 0.31),(2, 0.41),(-4, 0.48), (-10, 0.53),(-18, 0.57),(-27, 0.57),(-35, 0.57)], anchornumbers = True)
	title(f'Doubling the number of ETH-1 analyses', size = 9, weight = 'bold')

	data3 = [r.copy() for r in data]
	alt_Nominal_D47 = {
		'ETH-1': Nominal_D47['ETH-1'],
		'MERCK': mean([r['D47'] for r in data if r['Sample'] == 'MERCK']),
		'ETH-3': Nominal_D47['ETH-3'],
		}
	a, b, c, CM, rD47 = normalize(data3, alt_Nominal_D47)
	ax = subplot(224)
	errormap(data3, a, b, c, CM, alt_Nominal_D47, [(2, 0.62), (-40, 0.4)], anchornumbers = True)
	title(f'Using MERCK as an anchor instead of ETH-2', size = 9, weight = 'bold')
# 	ax.set_yticklabels(['']*len(ax.get_yticklabels()))
	ylabel('')
	yticks([])
	
	savefig('Fig1_properties_of_standardization_errors.pdf')
	close(fig)

def fig_allo_vs_auto_erors():
	out = {}
	data = read_csv(f'rawdata.csv')
	for r in data:
		r['Session'] = f"{r['Lab']}_{r['Session']}"
	sessions = sorted({r['Session'] for r in data})
	with open('commands.tex', 'a') as fid:
		fid.write(f'\\newcommand{{\\NInterCarbSessions}}{{{len(sessions)}}}\n')
		fid.write(f'\\newcommand{{\\NInterCarbAnalyses}}{{{len(data)}}}\n')
	for session in sessions:
		S = [r for r in data if r['Session'] == session]
		a, b, c, CM, rD47 = normalize(S, Nominal_D47)
		samples = sorted({r['Sample'] for r in S})
		unknowns = [s for s in samples if s not in Nominal_D47]
		for u in unknowns:
			if u not in out:
				out[u] = []
			U = [r for r in S if r['Sample'] == u]
			d47 = mean([r['d47'] for r in U])
			D47raw = mean([r['D47raw'] for r in U])
			autogenic_error = rD47 / len(U)**.5
			allogenic_error = total_error(a, b, c, CM, d47, D47raw, 0)			
			out[u] += [(autogenic_error, allogenic_error)]

	fig = figure(figsize = (6,4))
	subplots_adjust(left = .11, right = .99, bottom = .13, top = .97, hspace = 0.18, wspace = .2)
	
	for k,u in enumerate(out):
		ax = subplot(230 + [1,2,4,5][k])
		X, Y = zip(*(out[u]))
		X, Y = 1000*array(X), 1000*array(Y)
		kw = dict(ls = 'None', ms = 4, alpha = .75, mec = 'k', mfc = 'None', mew = 2/3, marker = 's')
# 		kw['marker'] = {'ETH-4':(3,0,0), 'IAEA-C1':(4,0,0), 'IAEA-C2':(4,0,45), 'MERCK':(3,0,180)}[u]
		plot(X, Y, **kw)
		x1, x2, y1, y2 = axis()
		emax = max(x2, y2)
		emax = 82
		axis([0, emax, 0, emax])
		if k > 1:
			xlabel('Autogenic error (σ$_u$, ppm)')
		if not k % 2:
			ylabel('Standardization\nerror (σ$_s$, ppm)')
		text(0.95, 0.95, u, va = 'top', ha = 'right', weight = 'bold', size = 10, transform = ax.transAxes)
		tiks = linspace(0,80,5)
		for e in tiks[1:]:
			gca().add_artist(
				Ellipse(
					xy = (0,0), width = 2*e, height = 2*e,
					lw = .5, fc = 'none', ec = 'k', alpha = .25, ls = '-', zorder = -100 )
				)
		plot([0,63], [0,63], 'k-', dashes = (8,3), lw = 0.75)
		xticks(tiks)
		yticks(tiks)

		ax = subplot(4,3, k*3 + 3)
		X, Y = zip(*(out[u]))
		X, Y = 1000*array(X), 1000*array(Y)
		R = (1 + (Y/X)**2)**.5
# 		print(u, mean(R))

		kw = dict(bins = linspace(1,4,25), histtype = 'stepfilled', color = 'k', alpha = .2, lw = 0)
# 		kw['marker'] = {'ETH-4':(3,0,0), 'IAEA-C1':(4,0,0), 'IAEA-C2':(4,0,45), 'MERCK':(3,0,180)}[u]
		hist(R, **kw)
		kw = dict(bins = linspace(1,4,25), histtype = 'step', color = 'k', alpha = 1, lw = 1)
		hist(R, **kw)
# 		x1, x2, y1, y2 = axis()
# 		emax = max(x2, y2)
# 		emax = 82
		ymax = axis()[-1] * 1.1
		axis([0.9, 4.1, 0, ymax])
		xticks([1,2,3,4])
		if k == 3:
			xlabel('$σ_{47}$ / $σ_{u}$')
		else:
		    ax.set_xticklabels( ['' for x in xticks()[0]] )
		yticks([])
		grid(alpha = .3)
# 		if not k % 2:
# 			ylabel('Standardization error (ppm)')
		if k < 4:
			text(0.94, 0.85, u, va = 'top', ha = 'right', weight = 'bold', size = 10, transform = ax.transAxes)
		else:
			text(0.05, 0.85, u, va = 'top', ha = 'left', weight = 'bold', size = 10, transform = ax.transAxes)

	savefig('Fig2_allo_vs_auto_errors.pdf')
	close(fig)

def montecarlo_normality(rng, S, KS, session, rD47 = 'None', Nmc = N_MONTECARLO):
	A = array([
		[ Nominal_D47[r['Sample']], r['d47'], 1 ]
		for r in S if r['Sample'] in Nominal_D47
		])
	Y = array([
		[ r['D47raw'] ]
		for r in S if r['Sample'] in Nominal_D47
		])

	CM = inv( A.T @ A )
	a0,b0,c0 = ( CM @ A.T @ Y )[:,0]

	for r in S :
		r['D47'] = ( r['D47raw'] - b0 * r['d47'] - c0)/a0

	if rD47 == 'None':
		samples = sorted({r['Sample'] for r in S})
		avgD47 = {
			sample: mean([r['D47'] for r in S if r['Sample'] == sample])
			for sample in samples
			}
		chi2 = sum([(r['D47'] - avgD47[r['Sample']])**2 for r in S])
		Ns = len(samples)
		Na = len(S)
		rD47 = sqrt(chi2/(Na-Ns))
	CM0 = (a0*rD47)**2 * CM
	
	auto_jiggle = rng.normal(loc = 0, scale = rD47, size = (len(S), Nmc))
	allo_jiggle = rng.normal(loc = 0, scale = rD47, size = (Y.size, Nmc))

	for k,r in enumerate(S) :
		r['auto_D47'] = ( r['D47raw'] - b0 * r['d47'] - c0)/a0 + auto_jiggle[k,:]

	Y = Y + allo_jiggle * a0
	a,b,c = ( CM @ A.T @ Y )

	for r in S :
		r['allo_D47'] = ( r['D47raw'] - b * r['d47'] - c ) / a

	for k,r in enumerate(S) :
		r['auto_and_allo_D47'] = ( r['D47raw'] - b * r['d47'] - c + auto_jiggle[k,:]*a0)/a

	samples = sorted({r['Sample'] for r in S})
	unknowns = [s for s in samples if s not in Nominal_D47]
	for u in unknowns:
# 		fig = figure()
		U = [r for r in S if r['Sample'] == u]
		d47 = mean([r['d47'] for r in U])
		D47raw = mean([r['D47raw'] for r in U])
		D47 = (D47raw - b0*d47 - c0)/a0
		autogenic_error = rD47 / len(U)**.5
		allogenic_error = total_error(a0, b0, c0, CM0, d47, D47raw, 0)
		both_errors = sqrt(autogenic_error**2 + allogenic_error**2)

		X = array([r['auto_D47'] for r in U]).mean(0)
		X.sort()
		Yo = arange(Nmc)/(Nmc-1)
		Yh = norm(D47, autogenic_error).cdf(X)
		KS['auto'] += [dict(
			pvalue = kstest(X, 'norm', (D47, autogenic_error), mode = 'asymp').pvalue,
			sample = u,
			session = session,
			X = X,
			Yo = Yo,
			Yh = Yh,
			D47 = D47,
			s_a = CM0[0,0]**.5,
			sigma = autogenic_error,
			)]

		X = array([r['allo_D47'] for r in U]).mean(0)
		X.sort()
		Yo = arange(Nmc)/(Nmc-1)
		Yh = norm(D47, allogenic_error).cdf(X)
		KS['allo'] += [dict(
			pvalue = kstest(X, 'norm', (D47, allogenic_error), mode = 'asymp').pvalue,
			sample = u,
			session = session,
			X = X,
			Yo = Yo,
			Yh = Yh,
			D47 = D47,
			s_a = CM0[0,0]**.5,
			sigma = allogenic_error,
			)]

		X = array([r['auto_and_allo_D47'] for r in U]).mean(0)
		X.sort()
		Yo = arange(Nmc)/(Nmc-1)
		Yh = norm(D47, both_errors).cdf(X)
		KS['both'] += [dict(
			pvalue = kstest(X, 'norm', (D47, both_errors), mode = 'asymp').pvalue,
			sample = u,
			session = session,
			X = X,
			Yo = Yo,
			Yh = Yh,
			D47 = D47,
			s_a = CM0[0,0]**.5,
			sigma = both_errors,
			)]
	

def test_normality(rD47 = 'None', ultext = '', lrtext = ''):
	rng = nprandom.default_rng(SEED)
# 	nprandom.seed('oof')
	data = read_csv(f'rawdata.csv')
	KS = dict(auto = [], allo = [], both = [])
	for r in data:
		r['Session'] = f"{r['Lab']}_{r['Session']}"
	sessions = sorted({r['Session'] for r in data})
	newfit_residuals = []
	for session in tqdm(sessions):
# 		print(f'{k+1}/{len(sessions)}')
		S = [r for r in data if r['Session'] == session]
		montecarlo_normality(rng, S, KS, session, rD47)

	plot([0,1], [0,1], 'k-', alpha = .25, lw = 1)
	axis([0,1,0,1])
	xlabel('K-S  p-value', labelpad = -10)
	if ultext == 'A':
		ylabel('Cumulative\ndistribution function', labelpad = -8)

	for which, color, label in [
		('allo', (1,.5,0), 'allogenic errors'),
		('both', (.25,.25,1), 'full errors'),
		('auto', (0,.75,.25), 'autogenic errors'),
		]:
		X = array([ks['pvalue'] for ks in KS[which]])
		X.sort()
		Y = arange(X.size)/(X.size-1)

		p = kstest(X, 'uniform').pvalue
		if p<1e-3:
			pstring = f'p < $10^{{{ceil(log10(p)):.0f}}}$'
		else:
			pstring = f"p = {p:.2f}" if p > 5e-3 else f"p = {p:.3f}"
# 		pstring = f"${p:.0e}}}$".replace('e-', '×10^{–') if p < 1e-3 else (f"{p:.2f}" if p > 5e-3 else f"{p:.3f}")
		if rD47 == 'None':
			with open('commands.tex', 'a') as fid:
				if which == 'auto':
					fid.write(f'\\newcommand{{\\pAutoNormality}}{{{pstring}}}\n')
				elif which == 'allo':
					fid.write(f'\\newcommand{{\\pAlloNormality}}{{{pstring}}}\n')
		plot(X, Y, '-', color = color, lw = 2,
			label = f"{label} ({pstring})"
			)
		f  = interpolate.interp1d(X, Y)
	
	text(.95, .05, lrtext, ha = 'right', transform = gca().transAxes)
	text(.03, .97, ultext, weight = 'bold', va = 'top', transform = gca().transAxes)
	xticks([0,1])
	yticks([0,1])

	legend(
		loc = 'upper center',
		bbox_to_anchor = (0.5, -.15),
		fontsize = 9,
		frameon = False,
		handlelength = 1.6,
		labelspacing = 0.4,
		)

	if rD47 == 'None':
	
		fig = figure()
		X = array([ks['D47'] for ks in KS['both']])
		sX = array([ks['sigma'] for ks in KS['both']])
		Y = array([mean(ks["X"]) for ks in KS['both']])
		sY = array([stdev(ks["X"]) for ks in KS['both']])
		
		plot((X-Y)/sX, (sX-sY)/sX, 'r+')
		xlabel('Scaled error in final Δ$_{47}$ value')
		ylabel('Scaled error in final Δ$_{47}$ SE')
# 		show()
		close(fig)
		with open('commands.tex', 'a') as fid:
			fid.write(f'\\newcommand{{\\worseNonGaussScalledErrorOnMean}}{{{max(abs((X-Y)/sX))*100:.0f}}}\n')
			fid.write(f'\\newcommand{{\\worseNonGaussScalledErrorOnSE}}{{{max(abs((sX-sY)/sX))*100:.0f}}}\n')
			fid.write(f'\\newcommand{{\\typicalNonGaussScalledErrorOnMean}}{{{mean(((X-Y)/sX)**2)**.5:.2f}}}\n')
			fid.write(f'\\newcommand{{\\typicalNonGaussScalledErrorOnSE}}{{{mean(((sX-sY)/sX)**2)**.5:.2f}}}\n')
	
		p = [ks['pvalue'] for ks in KS['allo']]
		k = p.index(min(p))
		ks = KS['both'][k]
# 		print(ks)
		fig = figure(figsize = (3.5,4.5))
		figure_margins(fig, 10, 2, 12, 2, 0, -0.15)

		subplot(311)
		bins = linspace(.55,.80,250+1)
		hist(ks['X'],
			bins = bins,
			histtype = 'stepfilled',
			color = 'r',
			alpha = 1/3,
			lw = 1,
			label = 'Observed',
			)
		plot(bins,
			norm(ks['D47'], ks['sigma']).pdf(bins)*ks['X'].size*(bins[1]-bins[0]),
			'k-', lw = 1, label = 'Gaussian')
		axis([ks['X'].min(), ks['X'].max(), None, None])
		yticks([])
		ylabel('Probability\ndistribution function', labelpad = 2)
# 		legend()

		subplot(212)
		f = interpolate.interp1d(ks['Yo'], ks['X'])
# 		hw = linspace(0,max(max['X']-ks['D47'], ks['D47'] - max['X'], 1000)
# 		conf = [for w in hw]
# 		print(f'      MC Δ47 95 % CL = {(f(.025) + f(.975))/2:.4f} ± {(-f(.025) + f(.975))/2:.4f} ‰')
		f = interpolate.interp1d(ks['Yh'], ks['X'])
# 		print(f'Gaussian Δ47 95 % CL = {(f(.025) + f(.975))/2:.4f} ± {(-f(.025) + f(.975))/2:.4f} ‰')
# 		print(f'MC mean Δ47 = {mean(ks["X"]):.4f} ‰ ({ks["D47"]}, {ks["sigma"]*1.96})')

		plot(ks['X'], ks['Yo'], 'r-', lw = 1, label = 'Monte Carlo')
		plot(ks['X'], ks['Yh'], 'k-', lw = 1, label = 'Gaussian')
# 		pv = f"${ks['pvalue']:.0e}}}$".replace('e-0', '\\times10^{–')
		pv = f"{ks['pvalue']:.4f}"
		with open('commands.tex', 'a') as fid:
			fid.write(f'\\newcommand{{\\worseMCpv}}{{{pv}}}\n')
			fid.write(f'\\newcommand{{\\worseMCmeanMC}}{{{mean(ks["X"]):.4f}}}\n')
			fid.write(f'\\newcommand{{\\worseMCmeanGauss}}{{{ks["D47"]:.4f}}}\n')
			fid.write(f'\\newcommand{{\\worseMCsigmaMC}}{{{stdev(ks["X"]):.4f}}}\n')
			fid.write(f'\\newcommand{{\\worseMCsigmaGauss}}{{{ks["sigma"]:.4f}}}\n')
			fid.write(f'\\newcommand{{\\worseMCmeanOffset}}{{{(ks["D47"] - mean(ks["X"]))/ks["sigma"]:.2f}}}\n')
			fid.write(f'\\newcommand{{\\worseMCsigmaOffset}}{{{(ks["sigma"] - stdev(ks["X"]))/ks["sigma"]:.2f}}}\n')
		text(.05, .95, f'p = {pv}', ha = 'left', va = 'top', transform = gca().transAxes)
		xlabel(f"Δ$_{{47}}$ of {ks['sample']} in {ks['session'].split('_')[1]} of {ks['session'].split('_')[0]}")
		ylabel('Cumulative\ndistribution function', labelpad = -8)
		axis([ks['X'].min(), ks['X'].max(), -.02, 1.02])
		yticks([0,1])
		legend(loc = 'lower right')

		savefig('Fig4_normality_worse_example.pdf')
		close(fig)
			

def fig_lmfit_effect():
	import D47crunch_snapshot as D47crunch
	D47crunch.D47data.Nominal_D47 = {
		'ETH-1':   D47crunch.D47data.Nominal_D47['ETH-1'],
		'ETH-2':   D47crunch.D47data.Nominal_D47['ETH-2'],
		'ETH-3':   D47crunch.D47data.Nominal_D47['ETH-3'],
		}

	data = read_csv(f'rawdata.csv')
	data = [r for r in data if r['Lab'] == 'Lab12']
	data = D47crunch.D47data(data)
	data.wg()
	data.crunch()

	fig = figure(figsize = (7,4))
	figure_margins(fig, 15, 5, 12, 6, .08, .18)

	for x in [0, 1]:
	
		if x:
			data.split_samples()
		data.standardize(method = 'pooled')

		for k, session in enumerate(['Session01', 'Session02', 'Session03', 'Session04']):
			subplot(245 + k - x*4)
			errormap(data.sessions[session]['data'],
				data.sessions[session]['a'],
				data.sessions[session]['b'],
				data.sessions[session]['c'],
				data.sessions[session]['CM'],
				Nominal_D47,
				addwidth = 3,
				manual = (
					[(-5, .4), (-15, .4), (-30, .4), (-40, .4)] if x
					else [(-12, 0.4), (-33, 0.4)]),
				)

			ax = gca()
# 			t = text(0.08, 0.92, session, transform = ax.transAxes, size = 8, va = 'top')
# 			t.set_path_effects([PathEffects.withStroke(linewidth=8, foreground='w')])
# 			t = text(0.08, 0.92, session, transform = ax.transAxes, size = 8, va = 'top')
# 			t.set_bbox(dict(facecolor='w', alpha=1, lw=0, zorder = 200))

			if k == 0:
				if x:
					txt = 'Session-independent standardization models (only taking anchors into account):'
				else:
					txt = 'Pooled standardization model (taking anchors and unknowns into account):'
				text(0, 1.04, txt, transform = ax.transAxes, size = 10, weight = 'bold')
			if x:
				ax.set_xticklabels(['' for t in ax.get_xticklabels()])
				xlabel('')
			if k:
				ax.set_yticklabels(['' for t in ax.get_yticklabels()])
				ylabel('')

	savefig('Fig5_benefits_of_pooled_std.pdf')
	close(fig)

def fig_aliquots(Naliq = 16, Nanchor1 = 16, Nanchor2 = 2, Nseries = 20, Nseries_replicates = 4):
	import D47crunch_snapshot as D47crunch
	
	fig = figure(figsize = (80/25.4, 150/25.4))
	subplots_adjust(0.28, 0.23, .98, .99, .1, .15)

	D = D47crunch.D47data()
	samples = [
		dict(Sample = 'ETH-1', N = Nseries_replicates),
		dict(Sample = 'ETH-2', N = Nseries_replicates),
		dict(Sample = 'ETH-3', N = Nseries_replicates),
		]
	for k in range(Nseries):
		samples.append(dict(
			Sample = f'FOO-{k}',
			d13C_VPDB = -5.,
			d18O_VPDB = -5.,
			D47 = 0.5 + sin(k/4)/40,
			N = Nseries_replicates
			))
	D.simulate(samples, rD47 = 0.010, seed = 1)
	D.standardize()
	D.plot_sessions()

	X = list(range(Nseries))
	Y = [D.samples[f'FOO-{k}']['D47'] for k in X]
	eY = [D.samples[f'FOO-{k}']['SE_D47']*D.t95 for k in X]
	Z = [y-Y[0] for y in Y]
	ZeZ = [D.sample_average([f'FOO-{k}','FOO-0'], [1, -1], normalize = False) for k in X]
	Z, eZ  =zip(*ZeZ)

	ax = subplot(411)
	errorbar(X, Y, eY, ls = 'None', marker = 'None', ecolor = 'k', elinewidth = 1, capthick = 1, capsize = 3)
	plot(X[0], Y[0], 'ko', mec = 'k', mew = 1, ms = 5)
	plot(X[1:], Y[1:], 'wo', mec = 'k', mew = 1, ms = 5)
	axhline(Y[0], color = 'k', lw = .75, alpha = .25, zorder = -10)
	xticks([])
	ylabel('Δ$_{47}$ (‰)')
	axis([-1, Nseries, None, None])
	y1, y2 = axis()[-2:]
	yticks(linspace(46,54,5)/100)
	text(.95, .9, 'A', size = 14, weight = 'bold', transform = ax.transAxes, va = 'top', ha = 'right')
	
	ax = subplot(412)
	errorbar(X, Z, eZ, ls = 'None', marker = 'None', ecolor = 'k', elinewidth = 1, capthick = 1, capsize = 3)
	plot(X[0], Z[0], 'ko', mec = 'k', mew = 1, ms = 5)
	plot(X[1:], Z[1:], 'wo', mec = 'k', mew = 1, ms = 5)
	xticks([])
	axis([-1, Nseries, None, None])
	y3, y4 = axis()[-2:]
	yticks(linspace(-4,4,5)/100)
	axis([-1, Nseries, (y3+y4-y2+y1)/2, (y3+y4+y2-y1)/2])
	axhline(0, color = 'k', lw = .75, alpha = .25, zorder = -10)
	ylabel('Δ$_{47}$ difference relative\nto first sample (‰)')
	text(.95, .9, 'B', size = 14, weight = 'bold', transform = ax.transAxes, va = 'top', ha = 'right')

	ax = subplot(212)

	for Nanchor, ls in [
		(Nanchor1, '--'),
		(Nanchor2, '-'),
		]:
		D = D47crunch.D47data()
		D.simulate([
			dict(Sample = 'ETH-1', N = Nanchor),
			dict(Sample = 'ETH-2', N = Nanchor),
			dict(Sample = 'ETH-3', N = Nanchor),
			dict(Sample = 'FOO', d13C_VPDB = -5., d18O_VPDB = -5., D47 = 0.5, N = Naliq),
			dict(Sample = 'BAR', d13C_VPDB = -5., d18O_VPDB = -5., D47 = 0.5, N = Naliq),
			], rD47 = 0.010, seed = 1)
		D.standardize()
	
		CM = array([
			[D.sample_D47_covar('FOO'), D.sample_D47_covar('FOO', 'BAR')],
			[D.sample_D47_covar('FOO', 'BAR'), D.sample_D47_covar('BAR')]
			])
		X = D.samples['FOO']['D47']
		sX = D.samples['FOO']['SE_D47']
		Y = D.samples['BAR']['D47']
		sY = D.samples['BAR']['SE_D47']

		xmin = min(X-3*sX, Y-3*sY)
		xmax = max(X+3*sX, Y+3*sY)
		w,h,r = cov_ellipse(CM)
		plot([], [], 'r-', lw = 1, ls = ls, label = f'Joint 95 % confidence region ({Naliq} replicates\nper unknown; {Nanchor} replicates per anchor)')
		ax.add_artist(
			Ellipse(
				xy = (X,Y), width = w, height = h, angle = r,
				lw = 1., fc = 'none', ec = 'r', ls = ls ))

	plot([xmin, xmax], [xmin, xmax], 'k-', lw = 0.75, dashes = (6,2,3,2))
	axis([xmin, xmax, xmin, xmax])
	legend(loc = 'center', bbox_to_anchor = (.35, -0.45), frameon = False, labelspacing = 1.5, fontsize = 8)
	xlabel('Δ$_{47}$ of first aliquot (‰)')
	ylabel('Δ$_{47}$ of second aliquot (‰)')
	text(.05, .95, 'C', size = 14, weight = 'bold', transform = ax.transAxes, va = 'top', ha = 'left')
	yticks(arange(48,52)/100)

	savefig('Fig6_compare_aliquots.pdf')
	close(fig)
	
def fig_montecarlo():

	figg = figure(figsize = (170/25.4,75/25.4))
	figure_margins(figg, 10, 5, 25, 3)

	subplot(131)
	print('Testing normality with observed Δ47 repeatabilities:')
	test_normality('None', 'A', f'N = ${N_MONTECARLO:.0e}}}$\nMonte Carlo noise\nbased on observed\nΔ$_{{47}}$ repeatabilities'.replace('e+0', '0^{'))
	subplot(132)
	print('Testing normality with 50 ppm Δ47 repeatability:')
	test_normality(0.05, 'B', f'N = ${N_MONTECARLO:.0e}}}$\nMonte Carlo\nnoise based on Δ$_{{47}}$\nrepeatability of 50 ppm'.replace('e+0', '0^{'))
	subplot(133)
	print('Testing normality with 5 ppmd Δ47 repeatability:')
	test_normality(0.005, 'C', f'N = ${N_MONTECARLO:.0e}}}$\nMonte Carlo\nnoise based on Δ$_{{47}}$\nrepeatability of 5 ppm'.replace('e+0', '0^{'))

	savefig('Fig7_normality.pdf')
	close(figg)

if __name__ == '__main__':

	Nominal_D47 = {}
	Nominal_D47['ETH-1'] = 0.2052
	Nominal_D47['ETH-2'] = 0.2085
	Nominal_D47['ETH-3'] = 0.6132
	# I-CDES values

	data = read_csv('rawdata.csv')

	with open('commands.tex', 'w') as fid:

		N_lab_session_sample = len({f"{r['Lab']}_{r['Session']}_{r['Sample']}" for r in data if r['Sample'] not in Nominal_D47})
		fid.write(f'\\newcommand{{\\Nlss}}{{{N_lab_session_sample}}}\n')

		N_MONTECARLO_str = f'{N_MONTECARLO:.0e}'.replace('e+0', '0\\hi{') + '}'
		fid.write(f'\\newcommand{{\\Nmc}}{{{N_MONTECARLO_str}}}\n')

	fig_properties_of_standardization_errors()
	fig_allo_vs_auto_erors()
	fig_lmfit_effect()
	fig_aliquots()
	fig_montecarlo()

	
