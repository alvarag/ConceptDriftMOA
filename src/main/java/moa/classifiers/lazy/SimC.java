/*
 * SimC.java
 * Copyright (C) 2016 Burgos University, Spain 
 * @author Álvar Arnaiz-González
 *     
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *     
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *     
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package moa.classifiers.lazy;

import java.util.LinkedList;
import java.util.TreeMap;

import weka.core.HVDM;
import weka.core.UpdateableDistanceFunction;
import weka.core.Utils;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

/**
 * Implementation of SimC. Presented in: Mena-Torres, D., & Aguilar-Ruiz, J. S.
 * (2014). A similarity-based approach for data stream classification. Expert
 * Systems with Applications, 41(9), 4224-4234.
 * <p>
 * This class is a wrapper to use the original code of the authors. The changes
 * made are minimum, only those necessary for MOA execution.<br> 
 * The original code is available at:
 * https://www.dropbox.com/s/s2t2ogaki1x1n4w/Weka.rar?dl=0
 * <p>
 * Valid options are:
 * k number of nearest neighbours to use <p>
 * w maximum size of the window <p>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160523
 */
public class SimC extends AbstractClassifier {

	private static final long serialVersionUID = -8315037677328629666L;

	public IntOption kOption = new IntOption("k", 'k', "The number of neighbors", 
	                                          1, 1, Integer.MAX_VALUE);

	public IntOption maxWinSizeOpt = new IntOption("maxWindowSize", 'w',
	                  "Maximum size of the window", 400, 1, Integer.MAX_VALUE);

	private int mInitTrain;

	@Override
	public boolean isRandomizable() {

		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		double[] dist = new double[inst.numClasses()];
		
		// First prediction, when any instance have arrived yet
		if (mInitTrain == 0)
			return dist;
		
		// Prediction.
		dist = distributionForInstance(inst);
		classifyInstance(inst, dist);
		
		return dist;
	}

	@Override
	public void resetLearningImpl() {
		DS = null;
		last100 = null;
		BI = null;
		MC = null;
		BienC = null;
		cantidad = 0;
		cantidadXgrupo = 0;
		numero_grupos = 0;
		matriz = new int[0][0];
		matriz_aux = new int[0][0];
		pos = 0;
		ubicacion_vec_mas_cercano = new LinkedList<Double>();
		lista_clases = new LinkedList<Double>();
		lista_streaming = new LinkedList<Integer>();
		acurracy_fols = new LinkedList<Double>();
		bien_clasificadas = new LinkedList<Double>();
		bien_clasf_total = 0;
		acc_fols = 0.0;
		cant_inst_fols = 0;
		cant_inst_bien_fols = 0;
		b1 = 0;
		b2 = 0;
		average = 0;
		inst_clasif_SCR = 0;
		T = 0;
		entropia = 0.2;
		contadorC_aux = 0;
		contadorMC_aux = 0;
		contador_total = 0;
		contador_streaming = 0;
		estruct = null;
		estruct_afuera = null;
		estruct_aux = null;
		
		mInitTrain = 0;
		distanceFunction = null;
		distanceFunction1 = null;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (numero_grupos == 0)
			numero_grupos = inst.numClasses();

		if (cantidadXgrupo == 0)
			cantidadXgrupo = (int) 100 / numero_grupos;

		if (estruct == null)
			estruct = new TreeMap<Double, TreeMap<Integer, Estructura>>();

		if (estruct_afuera == null)
			estruct_afuera = new TreeMap<Integer, Estructura>();

		if (estruct_aux == null)
			estruct_aux = new TreeMap<Double, TreeMap<Integer, Estructura>>();

		if (distanceFunction1 == null) {
			Instances insts = new Instances (inst.dataset(), 1);
			insts.add(inst);
			distanceFunction1 = new HVDM();
			distanceFunction1.setInstances(insts);
		}

		if (DS == null)
			DS = new Instances(inst.dataset(), 0);

		if (MC == null)
			MC = new Instances(inst.dataset(), 0);

		if (BienC == null)
			BienC = new Instances(inst.dataset(), 0);

		// Initialization with the first 100 instances.
		if (mInitTrain < 100) {
			mInitTrain++;

			try {
				estruct_afuera = estruct.get(inst.classValue());
				grupo_actual = new Instances(DS, 0);
				media_actual = null;
				edad_actual = 0;
				
				if (estruct_afuera == null) {
					grupo_actual.add(inst);
					media_actual = inst;
					edad_actual = 1;
					estruct_actual = new Estructura();
					estruct_actual.setGrupo(grupo_actual);
					estruct_actual.setMedia(media_actual);
					estruct_actual.setEdad(edad_actual);
					estruct_afuera = new TreeMap<Integer, Estructura>();
					estruct_afuera.put(0, estruct_actual);

					estruct.put(inst.classValue(), estruct_afuera);
					lista_clases.add(inst.classValue());
					cantidad++;
				} else {
					estruct_actual = estruct_afuera.get(estruct_afuera.firstKey());
					grupo_actual = estruct_actual.getGrupo();
					media_actual = estruct_actual.getMedia();
					edad_actual = estruct_actual.getEdad();

					if (cantidadXgrupo > grupo_actual.numInstances()) {
						grupo_actual.add(inst);
						estruct_actual.setGrupo(grupo_actual);

						media_actual = Calcular_Media_Inst(grupo_actual);
						estruct_actual.setMedia(media_actual);

						edad_actual = edad_actual + 1;
						estruct_actual.setEdad(edad_actual);

						estruct_afuera.put(estruct_afuera.firstKey(), estruct_actual);

						estruct.put(inst.classValue(), estruct_afuera);
						cantidad++;
					}
				}
				last100 = new Instances(inst.dataset(), 0);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		// Online update
		else {
			updateClassifier(inst);
		}
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		
	}

	private class Estructura {
		Instances grupo;

		Instance media;

		Integer edad;

		public void setEdad(Integer edad) {
			this.edad = edad;
		}

		public void setGrupo(Instances grupo) {
			this.grupo = grupo;
		}

		public void setMedia(Instance media) {
			this.media = media;
		}

		public Integer getEdad() {
			return edad;
		}

		public Instances getGrupo() {
			return grupo;
		}

		public Instance getMedia() {
			return media;
		}

		public Estructura(Instances data, Instance data_inst) {
			grupo = new Instances(data);
			media = (Instance) data_inst.copy();
			edad = 1;
		}

		public Estructura() {
		}
	}

	public Instances DS = null;

	public Instances last100 = null;

	/** Arreglo para la Base de Instancias */
	public Instances BI = null;

	public Instances MC = null;

	public Instances BienC = null;

	public int cantidad = 0;

	public int cantidadXgrupo = 0;

	public int numero_grupos = 0;

	public int matriz[][] = new int[0][0];

	public int matriz_aux[][] = new int[0][0];

	int pos = 0;

	public LinkedList<Double> ubicacion_vec_mas_cercano = new LinkedList<>();

	public LinkedList<Double> lista_clases = new LinkedList<>();

	public LinkedList<Integer> lista_streaming = new LinkedList<>();

	public LinkedList<Double> acurracy_fols = new LinkedList<>();

	public LinkedList<Double> bien_clasificadas = new LinkedList<>();

	Integer bien_clasf_total = 0;

	Double acc_fols = 0.0;

	Integer cant_inst_fols = 0;

	Integer cant_inst_bien_fols = 0;

	public double b1 = 0;

	public double b2 = 0;

	/** Variable para almacenar la cantidad de instancias bien clasificadas */
	public int average = 0;

	/** Variable para almacenar la cantidad de instancias clasificadas */
	// public int inst_clasif = 0;

	/**
	 * Variable para almacenar la cantidad de instancias clasificadas en el
	 * streaming clasification rate
	 */
	public int inst_clasif_SCR = 0;

	double T = 0;

	double entropia = 0.2;

	int contadorC_aux = 0;

	double contadorMC_aux = 0;

	protected TreeMap<Integer, Estructura> estruct_afuera = new TreeMap<Integer, Estructura>();

	// contador total de todas las instancias vistas
	protected Integer contador_total = 0;

	// contador total de todas las instancias vistas
	protected double contador_streaming = 0;

	// nueva con la estructura
	protected TreeMap<Double, TreeMap<Integer, Estructura>> estruct = new TreeMap<Double, TreeMap<Integer, Estructura>>();

	protected TreeMap<Double, TreeMap<Integer, Estructura>> estruct_aux = new TreeMap<Double, TreeMap<Integer, Estructura>>();

	Instances grupo_actual;

	Instance media_actual;

	Integer edad_actual;

	Estructura estruct_actual;

	protected double band;

	protected Instances neightbours;

//	protected UpdateableDistanceFunction distanceFunction1 = new HVDM();
	protected UpdateableDistanceFunction distanceFunction1;

	// protected DistanceFunction distanceFunction = new HVDM();
	protected UpdateableDistanceFunction distanceFunction;

	// public DistanceFunction getDistanceFunction() {
	public UpdateableDistanceFunction getDistanceFunction() {
		return distanceFunction;
	}

	// public void setDistanceFunction(DistanceFunction distanceFunction) {
	public void setDistanceFunction(UpdateableDistanceFunction distanceFunction) {
		this.distanceFunction = distanceFunction;
	}

	public UpdateableDistanceFunction getDistanceFunction1() {
		return distanceFunction1;
	}

	public void setDistanceFunction1(
			UpdateableDistanceFunction distanceFunction1) {
		this.distanceFunction1 = distanceFunction1;
	}

	// public void Actualizar_Contadores() {
	// contador_total++;
	//
	// //
	// // if (contador_total == 19999) {
	// // // String nombreArchivo = "Mal_Clasificados"; // Aqui se le asigna el
	// nombre y
	// // FileWriter fw = null; // la extension al archivo
	// // try {
	// // fw = new FileWriter("Mal_Clasificados");
	// // BufferedWriter bw = new BufferedWriter(fw);
	// // PrintWriter salArch = new PrintWriter(bw);
	// // salArch.print(MC.toString());
	// // salArch.println();
	// // salArch.close();
	// // } catch (IOException ex) {
	// // }
	// // fw = null; // la extension al archivo
	// // try {
	// // fw = new FileWriter("Bien_Clasificados");
	// // BufferedWriter bw = new BufferedWriter(fw);
	// // PrintWriter salArch = new PrintWriter(bw);
	// // salArch.print(BienC.toString());
	// // salArch.println();
	// // salArch.close();
	// // } catch (IOException ex) {
	// // }
	// // fw = null; // la extension al archivo
	// // try {
	// // fw = new FileWriter("Estructura_f");
	// // BufferedWriter bw = new BufferedWriter(fw);
	// // PrintWriter salArch = new PrintWriter(bw);
	// // salArch.print(estruct.toString());
	// //
	// // FileWriter fw_aux = new FileWriter("total.arff");
	// // BufferedWriter bw_aux = new BufferedWriter(fw_aux);
	// // PrintWriter salArch_aux = new PrintWriter(bw_aux);
	// //
	// // FileWriter fw_medias = new FileWriter("medias.arff");
	// // BufferedWriter bw_medias = new BufferedWriter(fw_medias);
	// // PrintWriter salArch_medias = new PrintWriter(bw_medias);
	// //
	// // for (int i = 0; i < lista_clases.size(); i++) {
	// // // salArch.print(estruct.get(lista_clases.get(i)).toString());
	// // int it = 0;
	// // for (int ii = 0; ii <
	// estruct.get(lista_clases.get(i)).keySet().size(); ii++) {
	// // Integer[] array_it =
	// estruct.get(lista_clases.get(i)).keySet().toArray(
	// // new Integer[0]);
	// // it = array_it[ii];
	// // salArch.println();
	// //// salArch.print("grupo: " +
	// estruct.get(lista_clases.get(i)).get(it).getGrupo().toString());
	// //// salArch.println();
	// // for(int y = 0;
	// y<estruct.get(lista_clases.get(i)).get(it).getGrupo().numInstances();
	// y++){
	// //
	// salArch.print(estruct.get(lista_clases.get(i)).get(it).getGrupo().instance(y).toString()
	// +" weight: "+estruct.get(lista_clases.get(i)).get(it).getGrupo().instance(y).weight());
	// // salArch.println();
	// // }
	// ////
	// salArch_aux.print(estruct.get(lista_clases.get(i)).get(it).getGrupo().toString());
	// // salArch.println();
	// //
	// //
	// salArch_medias.print(estruct.get(lista_clases.get(i)).get(it).getMedia().toString());
	// // salArch_medias.println();
	// //
	// // salArch.print("media: " +
	// estruct.get(lista_clases.get(i)).get(it).getMedia().toString());
	// // salArch.println();
	// // salArch.print("edad: " +
	// estruct.get(lista_clases.get(i)).get(it).getEdad().toString());
	// //
	// // }
	// // }
	// // salArch_aux.println();
	// // salArch_aux.close();
	// //
	// // salArch_medias.println();
	// // salArch_medias.close();
	// //
	// // salArch.println();
	// // salArch.close();
	// // } catch (IOException ex) {
	// // }
	// //
	// // }
	// }

	public LinkedList<Double> Buscar_Cluster_Mas_Cercano(Instance value)
			throws Exception {
		LinkedList<Double> res = new LinkedList<Double>();
		TreeMap<Integer, Estructura> tm_estruct_afuera = new TreeMap<Integer, Estructura>();
		Estructura tm_estruct = new Estructura();
		Instance tm_media = null;
		double dist = 1000;
		double c = -1;
		Integer g = -1;
		for (int i = 0; i < lista_clases.size(); i++) {
			// if (lista_clases.get(i) != value.classValue()) {
			tm_estruct_afuera = estruct.get(lista_clases.get(i));
			int it = 0;
			for (int ii = 0; ii < tm_estruct_afuera.keySet().size(); ii++) {
				Integer[] array_it = tm_estruct_afuera.keySet().toArray(
						new Integer[0]);
				it = array_it[ii];
				tm_estruct = tm_estruct_afuera.get(it);
				tm_media = tm_estruct.getMedia();
				if (dist > (double) distanceFunction1.distance(tm_media, value,
						Double.MIN_VALUE)) { //
					dist = (double) distanceFunction1.distance(tm_media, value,
							Double.MIN_VALUE);
					c = lista_clases.get(i);
					g = it;
				}
			}
			// }

		}
		res.add(dist);
		res.add(c);
		res.add(g.doubleValue());
		return res;
	}

	public LinkedList<Double> Buscar_Cercanias_aCluster_Misma_Clase(
			Instance value) throws Exception {
		LinkedList<Double> res = new LinkedList<Double>();
		TreeMap<Integer, Estructura> tm_estruct_afuera = new TreeMap<Integer, Estructura>();
		Estructura tm_estruct = new Estructura();
		Instance tm_media = null;
		double dist = 1000;
		double c = -1;
		Integer g = -1;
		for (int i = 0; i < lista_clases.size(); i++) {
			if (lista_clases.get(i) == value.classValue()) {
				tm_estruct_afuera = estruct.get(lista_clases.get(i));
				int it = 0;
				for (int ii = 0; ii < tm_estruct_afuera.keySet().size(); ii++) {
					Integer[] array_it = tm_estruct_afuera.keySet().toArray(
							new Integer[0]);
					it = array_it[ii];
					tm_estruct = tm_estruct_afuera.get(it);
					tm_media = tm_estruct.getMedia();
					double d = distanceFunction1.distance(tm_media, value);
					if (dist > distanceFunction1.distance(tm_media, value)) { //
						dist = distanceFunction1.distance(tm_media, value);
						c = value.classValue();
						g = ii;
					}
				}
			}

		}

		if (dist == 1000) {
			c = value.classValue();
			g = tm_estruct_afuera.firstKey();
		}

		res.add(dist);
		res.add(c);
		res.add(g.doubleValue());
		return res;
	}

	public Instance Calcular_Media_Inst(Instances aux) {
		Instance v = (Instance) aux.instance(0).copy();
		for (int j = 0; j < aux.numAttributes(); j++) {
			if (j != aux.classIndex()) {
//				double a = aux.meanOrMode(j);
				double a = meanOrMode(aux, j);
				v.setValue(j, a);
			}
		}
		
		return v;
	}

	public void Crear_Cluster(Instance value) throws Exception {
		TreeMap<Integer, Estructura> tm_afuera = new TreeMap<Integer, Estructura>();
		Estructura tm_estruct = new Estructura();
		Instances tm_grupo = new Instances(DS, 0);
		tm_afuera = estruct.get(value.classValue());
		int k = 0;
		if (tm_afuera != null) {
			if (tm_afuera.keySet().isEmpty()) {
				k = 0;
			} else {
				k = tm_afuera.lastKey() + 1;
			}
			tm_grupo.add(value);
			tm_estruct.setGrupo(tm_grupo);
			tm_estruct.setMedia(value);
			tm_estruct.setEdad(1);

			tm_afuera.put(k, tm_estruct);

			estruct.put(value.classValue(), tm_afuera);
		} else {
			tm_grupo = new Instances(DS, 0);
			tm_grupo.add(value);
			tm_afuera = new TreeMap<Integer, Estructura>();

			tm_estruct.setGrupo(tm_grupo);
			tm_estruct.setMedia(value);
			tm_estruct.setEdad(1);

			tm_afuera.put(k, tm_estruct);

			estruct.put(value.classValue(), tm_afuera);

			lista_clases.add(value.classValue());

		}
		if (!Eliminar_Ruido(value.classValue())) {
			Eliminar_Elemento_Menos_Util_Clase(value.classValue());
		}
	}

	private boolean Eliminar_Ruido(double cl) {
		TreeMap<Integer, Estructura> tm_afuera = new TreeMap<Integer, Estructura>();
		Estructura tm_estruct = new Estructura();
		tm_afuera = estruct.get(cl);
		int pos_eliminar = -1;
		int it = 0;
		for (int i = 0; i < tm_afuera.size(); i++) {
			Integer[] array_it = tm_afuera.keySet().toArray(new Integer[0]);
			it = array_it[i];
			tm_estruct = tm_afuera.get(it);
			if (tm_estruct.getEdad() == 1 && it != tm_afuera.lastKey()) { //
				pos_eliminar = it;
				break;
			}
		}
		if (pos_eliminar != -1) {
			for (int i = 0; i < tm_afuera.get(pos_eliminar).getGrupo()
					.numInstances(); i++) {
				distanceFunction1.remove(tm_afuera.get(pos_eliminar).getGrupo()
						.instance(i));
				cantidad--;
			}

			tm_afuera.remove(pos_eliminar);
			estruct.put(cl, tm_afuera);
			numero_grupos--;
//			cantidadXgrupo = (int) maxWindowSize / numero_grupos;
			cantidadXgrupo = (int) maxWinSizeOpt.getValue() / numero_grupos;
			return true;
		} else {
			return false;
		}
	}

	private void Eliminar_Grupo(double cl) throws Exception {
		TreeMap<Integer, Estructura> tm_afuera = new TreeMap<Integer, Estructura>();
		Estructura tm_estruct = new Estructura();
		tm_afuera = estruct.get(cl);
		int pos_eliminar = -1;
		int it = 0;
		for (int i = 0; i < tm_afuera.size(); i++) {
			Integer[] array_it = tm_afuera.keySet().toArray(new Integer[0]);
			it = array_it[i];
			tm_estruct = tm_afuera.get(it);
			if (tm_estruct.getGrupo().numInstances() < 2) { // //Aqui cambiar
															// por <
				pos_eliminar = it;
				// cant_elim = tm_estruct.getGrupo().numInstances();
				// tm_edad= tm_estruct.getEdad();
				break;
			}
		}
		if (pos_eliminar != -1) {
			for (int i = 0; i < tm_afuera.get(pos_eliminar).getGrupo()
					.numInstances(); i++) {
				distanceFunction1.remove(tm_afuera.get(pos_eliminar).getGrupo()
						.instance(i));
				cantidad--;
			}

			tm_afuera.remove(pos_eliminar);
			estruct.put(cl, tm_afuera);

			numero_grupos--;
//			cantidadXgrupo = (int) maxWindowSize / numero_grupos;
			cantidadXgrupo = (int) maxWinSizeOpt.getValue() / numero_grupos;
		}
	}

	public void buildClassifier(Instances data) throws Exception {
		// maxWindowSize = (int)(Math.log(data.numAttributes())*
		// data.numClasses()*10);
		numero_grupos = data.numClasses();
		cantidad = 0;
		cantidadXgrupo = (int) 100 / numero_grupos; // (int) maxWindowSize
													// /numero_grupos;
		estruct = new TreeMap<Double, TreeMap<Integer, Estructura>>();
		estruct_afuera = new TreeMap<Integer, Estructura>();
		distanceFunction1.setInstances(data);
		DS = new Instances(data, 0);
		MC = new Instances(data, 0);
		BienC = new Instances(data, 0);
		for (int i = 0; i < data.numInstances(); i++) {
			try {
				estruct_afuera = estruct.get(data.instance(i).classValue());
				grupo_actual = new Instances(DS, 0);
				media_actual = null;
				edad_actual = 0;
				// si no tengo instancias de esa clase en BC
				// crear una lista de medias, y tomar la posicion como llave
				// para el
				// tm_afuera
				if (estruct_afuera == null) {
					grupo_actual.add(data.instance(i));
					media_actual = data.instance(i);
					edad_actual = 1;
					estruct_actual = new Estructura();
					estruct_actual.setGrupo(grupo_actual);
					estruct_actual.setMedia(media_actual);
					estruct_actual.setEdad(edad_actual);
					estruct_afuera = new TreeMap<Integer, Estructura>();
					estruct_afuera.put(0, estruct_actual);

					estruct.put(data.instance(i).classValue(), estruct_afuera);
					lista_clases.add(data.instance(i).classValue());
					cantidad++;
				} else {
					estruct_actual = estruct_afuera.get(estruct_afuera
							.firstKey());
					grupo_actual = estruct_actual.getGrupo();
					media_actual = estruct_actual.getMedia();
					edad_actual = estruct_actual.getEdad();

					if (cantidadXgrupo > grupo_actual.numInstances()) {
						grupo_actual.add(data.instance(i));
						estruct_actual.setGrupo(grupo_actual);

						media_actual = Calcular_Media_Inst(grupo_actual);
						estruct_actual.setMedia(media_actual);

						edad_actual = edad_actual + 1;
						estruct_actual.setEdad(edad_actual);

						estruct_afuera.put(estruct_afuera.firstKey(),
								estruct_actual);

						estruct.put(data.instance(i).classValue(),
								estruct_afuera);
						cantidad++;
					}
				}
				last100 = new Instances(data, 0);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public void like_buildClassifier(Instances data) throws Exception {
		// maxWindowSize = (int)(Math.log(data.numAttributes())*
		// data.numClasses()*10);
		numero_grupos = data.numClasses();
		cantidad = 0; // (int) maxWindowSize /numero_grupos;
		cantidadXgrupo = (int) 100 / numero_grupos;
		estruct = new TreeMap<Double, TreeMap<Integer, Estructura>>();
		estruct_afuera = new TreeMap<Integer, Estructura>();
		distanceFunction1 = new HVDM();
		distanceFunction1.setInstances(data);
		DS = new Instances(data, 0);
		MC = new Instances(data, 0);
		BienC = new Instances(data, 0);
		lista_clases = new LinkedList<Double>();
		for (int i = 0; i < data.numInstances(); i++) {
			try {
				estruct_afuera = estruct.get(data.instance(i).classValue());
				grupo_actual = new Instances(DS, 0);
				media_actual = null;
				edad_actual = 0;
				// si no tengo instancias de esa clase en BC
				// crear una lista de medias, y tomar la posicion como llave
				// para el
				// tm_afuera
				if (estruct_afuera == null) {
					grupo_actual.add(data.instance(i));
					media_actual = data.instance(i);
					edad_actual = 1;
					estruct_actual = new Estructura();
					estruct_actual.setGrupo(grupo_actual);
					estruct_actual.setMedia(media_actual);
					estruct_actual.setEdad(edad_actual);
					estruct_afuera = new TreeMap<Integer, Estructura>();
					estruct_afuera.put(0, estruct_actual);

					estruct.put(data.instance(i).classValue(), estruct_afuera);
					lista_clases.add(data.instance(i).classValue());
					cantidad++;

				} else {
					estruct_actual = estruct_afuera.get(estruct_afuera
							.firstKey());
					grupo_actual = estruct_actual.getGrupo();
					media_actual = estruct_actual.getMedia();
					edad_actual = estruct_actual.getEdad();

					if (cantidadXgrupo > grupo_actual.numInstances()) {
						grupo_actual.add(data.instance(i));
						estruct_actual.setGrupo(grupo_actual);

						media_actual = Calcular_Media_Inst(grupo_actual);
						estruct_actual.setMedia(media_actual);

						edad_actual = edad_actual + 1;
						estruct_actual.setEdad(edad_actual);

						estruct_afuera.put(estruct_afuera.firstKey(),
								estruct_actual);

						estruct.put(data.instance(i).classValue(),
								estruct_afuera);
						cantidad++;
					}
				}
				// Actualizar_Contadores();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	private void update_good(Instance instance) throws Exception {
		// cantidad = (int) maxWindowSize / numero_grupos;
		cantidad++;
		distanceFunction1.add(instance);
		estruct_afuera = estruct.get(instance.classValue());
		grupo_actual = new Instances(DS, 0);
		media_actual = null;
		edad_actual = 0;

		LinkedList<Double> cercania = Buscar_Cluster_Mas_Cercano(instance);
		double clase = cercania.get(1).doubleValue();
		Integer grupo = cercania.get(2).intValue();
		if (instance.classValue() == clase) {
			estruct_afuera = estruct.get(clase);
			estruct_actual = estruct_afuera.get(grupo);
			grupo_actual = estruct_actual.getGrupo();
			media_actual = estruct_actual.getMedia();
			edad_actual = estruct_actual.getEdad();
//			if (cantidad < maxWindowSize
			if (cantidad < maxWinSizeOpt.getValue()
					&& grupo_actual.numInstances() < cantidadXgrupo) { // grupo_actual.numInstances()
																		// <
				grupo_actual.add(instance);
				media_actual = Calcular_Media_Inst(grupo_actual);
				edad_actual++;
				estruct_actual.setGrupo(grupo_actual);
				estruct_actual.setMedia(media_actual);
				estruct_actual.setEdad(edad_actual);
				estruct_afuera.put(grupo, estruct_actual);
				estruct.put(clase, estruct_afuera);
			} else {
				if (grupo_actual.numInstances() < cantidadXgrupo) {
					Eliminar_Elemento_Menos_Util_Grupo_Mas_Antiguo(clase, grupo);
				} else {
					grupo_actual = Eliminar_Elemento_Menos_Util_Grupo(grupo_actual);
				}
				try {
					estruct_afuera = estruct.get(clase);
					estruct_actual = estruct_afuera.get(grupo);
					grupo_actual = estruct_actual.getGrupo();
					edad_actual = estruct_actual.getEdad();

					grupo_actual.add(instance);
					media_actual = Calcular_Media_Inst(grupo_actual);
					edad_actual++;

					estruct_actual.setGrupo(grupo_actual);
					estruct_actual.setMedia(media_actual);
					estruct_actual.setEdad(edad_actual);
					estruct_afuera.put(grupo, estruct_actual);
					estruct.put(clase, estruct_afuera);
				} catch (Exception e) {
					numero_grupos++;
//					cantidadXgrupo = (int) maxWindowSize / numero_grupos;
					cantidadXgrupo = (int) maxWinSizeOpt.getValue() / numero_grupos;
					Crear_Cluster(instance);
				}
			}

		} else {

			clase = instance.classValue();
			grupo = ubicacion_vec_mas_cercano.get(1).intValue();

			estruct_afuera = estruct.get(clase);
			estruct_actual = estruct_afuera.get(grupo);
			grupo_actual = estruct_actual.getGrupo();
			media_actual = estruct_actual.getMedia();
			edad_actual = estruct_actual.getEdad();
//			if (cantidad < maxWindowSize
			if (cantidad < maxWinSizeOpt.getValue()
					&& grupo_actual.numInstances() < cantidadXgrupo) { // grupo_actual.numInstances()
																		// >
																		// cantidad
				grupo_actual.add(instance);
				media_actual = Calcular_Media_Inst(grupo_actual);
				edad_actual++;

				estruct_actual.setGrupo(grupo_actual);
				estruct_actual.setMedia(media_actual);
				estruct_actual.setEdad(edad_actual);
				estruct_afuera.put(grupo, estruct_actual);
				estruct.put(clase, estruct_afuera);

			} else {
				Eliminar_Elemento_Menos_Util_Grupo_Mas_Antiguo(clase, grupo);

				estruct_afuera = estruct.get(clase);
				estruct_actual = estruct_afuera.get(grupo);
				grupo_actual = estruct_actual.getGrupo();
				edad_actual = estruct_actual.getEdad();

				grupo_actual.add(instance);
				media_actual = Calcular_Media_Inst(grupo_actual);
				edad_actual++;
				estruct_actual.setGrupo(grupo_actual);
				estruct_actual.setMedia(media_actual);
				estruct_actual.setEdad(edad_actual);
				estruct_afuera.put(grupo, estruct_actual);
				estruct.put(clase, estruct_afuera);

			}
		}
	}

	private void update_bad(Instance instance) throws Exception {
		cantidad++; // cantidad = (int) maxWindowSize / numero_grupos;
		distanceFunction1.add(instance);
		estruct_afuera = estruct.get(instance.classValue());
		grupo_actual = new Instances(DS, 0);
		media_actual = null;
		edad_actual = 0;
		// si no tengo instancias de esa clase en BC
		// crear una lista de medias, y tomar la posicion como llave para el
		// tm_afuera
		if (estruct_afuera == null || estruct_afuera.size() == 0) {
			grupo_actual.add(instance);
			media_actual = instance;
			edad_actual = 1;
			estruct_actual = new Estructura();
			estruct_actual.setGrupo(grupo_actual);
			estruct_actual.setMedia(media_actual);
			estruct_actual.setEdad(edad_actual);
			estruct_afuera = new TreeMap<Integer, Estructura>();
			estruct_afuera.put(0, estruct_actual);

			estruct.put(instance.classValue(), estruct_afuera);
			lista_clases.add(instance.classValue());
			numero_grupos++;
//			cantidadXgrupo = (int) maxWindowSize / numero_grupos;
//			if (cantidad >= maxWindowSize) {
			cantidadXgrupo = (int) maxWinSizeOpt.getValue() / numero_grupos;
			if (cantidad >= maxWinSizeOpt.getValue()) {
				if (!Eliminar_Ruido(lista_clases.getFirst())) {
					Eliminar_Elemento_Menos_Util_Clase(instance.classValue());
				}
			}

		} else {
			LinkedList<Double> cercania = Buscar_Cluster_Mas_Cercano(instance);
			if (cercania.size() != 0) {
				double clase = cercania.get(1).doubleValue();
				Integer grupo = cercania.get(2).intValue();
				if (instance.classValue() == clase) {
					estruct_afuera = estruct.get(clase);
					estruct_actual = estruct_afuera.get(grupo);
					grupo_actual = estruct_actual.getGrupo();
					media_actual = estruct_actual.getMedia();
					edad_actual = estruct_actual.getEdad();
//					if (cantidad < maxWindowSize
					if (cantidad < maxWinSizeOpt.getValue()
							&& grupo_actual.numInstances() < cantidadXgrupo) { // grupo_actual.numInstances()
																				// <
																				// cantidad
						grupo_actual.add(instance);
						media_actual = Calcular_Media_Inst(grupo_actual);
						edad_actual++;

						estruct_actual.setGrupo(grupo_actual);
						estruct_actual.setMedia(media_actual);
						estruct_actual.setEdad(edad_actual);
						estruct_afuera.put(grupo, estruct_actual);
						estruct.put(clase, estruct_afuera);
					} else {
						if (grupo_actual.numInstances() < cantidadXgrupo) {
							Eliminar_Elemento_Menos_Util_Grupo_Mas_Antiguo(
									clase, grupo);
						} else {
							grupo_actual = Eliminar_Elemento_Menos_Util_Grupo(grupo_actual);
						}
						try {
							estruct_afuera = estruct.get(clase);
							estruct_actual = estruct_afuera.get(grupo);
							grupo_actual = estruct_actual.getGrupo();
							edad_actual = estruct_actual.getEdad();

							grupo_actual.add(instance);
							media_actual = Calcular_Media_Inst(grupo_actual);
							edad_actual++;

							estruct_actual.setGrupo(grupo_actual);
							estruct_actual.setMedia(media_actual);
							estruct_actual.setEdad(edad_actual);
							estruct_afuera.put(grupo, estruct_actual);
							estruct.put(clase, estruct_afuera);
						} catch (Exception e) {
							numero_grupos++;
//							cantidadXgrupo = (int) maxWindowSize
							cantidadXgrupo = (int) maxWinSizeOpt.getValue()
									/ numero_grupos;
							Crear_Cluster(instance);
						}
					}
				} else {
					try {
						Eliminar_Elemento_Menos_Util_Grupo_Mas_Antiguo(clase,
								grupo);
					} catch (Exception e) {
					}
					numero_grupos++;
//					cantidadXgrupo = (int) maxWindowSize / numero_grupos;
					cantidadXgrupo = (int) maxWinSizeOpt.getValue() / numero_grupos;
					Crear_Cluster(instance);

				}
			} else {
				numero_grupos++;
//				cantidadXgrupo = (int) maxWindowSize / numero_grupos;
				cantidadXgrupo = (int) maxWinSizeOpt.getValue() / numero_grupos;
				Crear_Cluster(instance);
			}
		}

		// Actualizar_Contadores();
	}

	private void update_with_concept_change() throws Exception {
		estruct = new TreeMap<Double, TreeMap<Integer, Estructura>>();
		like_buildClassifier(last100);
		lista_streaming = new LinkedList<Integer>();
		contadorC_aux = 0;
	}

	// private void Eliminar_Grupo_Mas_Antiguo(double classValue) {
	// TreeMap<Integer, Estructura> tm_afuera = new TreeMap<Integer,
	// Estructura>();
	// Estructura tm_estruct = new Estructura();
	// Instances tm_group = new Instances(DS,0);
	// double cl= 0;
	// for(int i=0; i<lista_clases.size();i++){
	// if(classValue!= lista_clases.get(i)){
	// tm_afuera = estruct.get(lista_clases.get(i));
	// cl=lista_clases.get(i);
	// break;
	// }
	// }
	// tm_estruct=tm_afuera.get(tm_afuera.firstKey());
	// tm_group = tm_estruct.getGrupo();
	// for(int i=0;i<tm_group.numInstances();i++){
	// distanceFunction1.remove(tm_group.instance(i));
	// }
	// numero_grupos--;
	// tm_afuera.remove(tm_afuera.firstKey());
	// estruct.put(cl, tm_afuera);
	// }

	// public LinkedList Eliminar_Elemento_Menos_Util_Union_2Grupos(Instances
	// grupo_cluster, Instances grupo_vec) {
	// LinkedList resultado = new LinkedList();
	// int eliminar = 0;
	// int tipo = -1;
	// double w = grupo_cluster.firstInstance().weight();
	// if (grupo_cluster.numInstances() >1) {
	// for (int i = 1; i < grupo_cluster.numInstances(); i++) {
	// if (w > grupo_cluster.instance(i).weight()) {
	// w = grupo_cluster.instance(i).weight();
	// eliminar = i;
	// tipo=0;
	// }
	// }
	// }
	//
	// if (grupo_vec.numInstances() > 1) {
	// for (int i = 1; i < grupo_vec.numInstances(); i++) {
	// if (w > grupo_vec.instance(i).weight()) {
	// w = grupo_vec.instance(i).weight();
	// eliminar = i;
	// tipo=1;
	// }
	// }
	// }
	//
	// if(tipo!=-1 && tipo==0){
	// distanceFunction1.remove(grupo_cluster.instance(eliminar));
	// grupo_cluster.delete(eliminar);
	// resultado.add(tipo);
	// resultado.add(grupo_cluster);
	// } else
	// if(tipo!=-1 && tipo==1){
	// distanceFunction1.remove(grupo_vec.instance(eliminar));
	// grupo_vec.delete(eliminar);
	// resultado.add(tipo);
	// resultado.add(grupo_vec);
	// }
	//
	// return resultado;
	//
	// }

	public Instances Eliminar_Elemento_Menos_Util_Grupo(Instances grupo)
			throws Exception {
		int eliminar = 0;
		// double w = grupo.firstInstance().weight();
		double w = grupo.instance(0).weight();
		if (grupo.numInstances() > 1) {
			for (int i = 1; i < grupo.numInstances(); i++) {
				if (w > grupo.instance(i).weight()) {
					w = grupo.instance(i).weight();
					eliminar = i;
				}
			}
			distanceFunction1.remove(grupo.instance(eliminar));
			grupo.delete(eliminar);
			cantidad--;
			return grupo;
		} else {
			// Eliminar_Grupo(grupo.firstInstance().classValue());
			Eliminar_Grupo(grupo.instance(0).classValue());
			return null;
		}

	}

	public void Eliminar_Elemento_Menos_Util_Clase(double clase)
			throws Exception {
		int eliminar = 0;
		TreeMap<Integer, Estructura> afuera = new TreeMap<Integer, Estructura>();
		afuera = estruct.get(clase);
		Estructura actual = new Estructura();
		actual = afuera.get(afuera.firstKey());
		Instances grupo = actual.getGrupo();
		// double w = grupo.firstInstance().weight();
		double w = grupo.instance(0).weight();

		if (grupo.numInstances() > 1) {
			for (int i = 1; i < grupo.numInstances(); i++) {
				if (1 == grupo.instance(i).weight()) {
					// w = grupo.instance(i).weight();
					eliminar = i;
					break;
				}
			}
			distanceFunction1.remove(grupo.instance(eliminar));
			cantidad--;
			grupo.delete(eliminar);
			Instance media = Calcular_Media_Inst(grupo);
			actual.setGrupo(grupo);
			actual.setMedia(media);
			afuera.put(afuera.firstKey(), actual);
			estruct.put(clase, afuera);
		} else {
			// Eliminar_Grupo(grupo.firstInstance().classValue());
			Eliminar_Grupo(grupo.instance(0).classValue());
		}
	}

	public void Eliminar_Elemento_Menos_Util_Grupo_Mas_Antiguo(Double c,
			Integer g) throws Exception {
		// buscar grupo mas antiguo (grupo de mayor edad)
		TreeMap<Integer, Estructura> tm_afuera = new TreeMap<Integer, Estructura>();
		Estructura tm_estruct = new Estructura();
		tm_afuera = estruct.get(c);
		int tm_edad = -1; // tm_afuera.get(tm_afuera.firstKey()).getEdad();
		int pos_eliminar = -1;// tm_afuera.firstKey();
		int it = 0;
		Integer[] array_it = tm_afuera.keySet().toArray(new Integer[0]);
		if (tm_afuera.size() > 1) {
			for (int i = 0; i < tm_afuera.size(); i++) {
				it = array_it[i];
				tm_estruct = tm_afuera.get(it);
				if (tm_estruct.getEdad() > tm_edad && it != g) { // //Aqui
																	// cambiar
																	// por <
					pos_eliminar = it;
					tm_edad = tm_estruct.getEdad();
				}
			}
		} else {
			pos_eliminar = tm_afuera.firstKey();
		}

		// buscar elemento menos util
		int eliminar = 0;
		// double w =
		// tm_afuera.get(pos_eliminar).getGrupo().firstInstance().weight();
		double w = tm_afuera.get(pos_eliminar).getGrupo().instance(0).weight();
		if (tm_afuera.get(pos_eliminar).getGrupo().numInstances() > 1) {
			for (int i = 1; i < tm_afuera.get(pos_eliminar).getGrupo()
					.numInstances(); i++) {
				if (w > tm_afuera.get(pos_eliminar).getGrupo().instance(i)
						.weight()) {
					w = tm_afuera.get(pos_eliminar).getGrupo().instance(i)
							.weight();
					eliminar = i;
				}
			}
			distanceFunction1.remove(tm_afuera.get(pos_eliminar).getGrupo()
					.instance(eliminar));
			cantidad--;
			tm_afuera.get(pos_eliminar).getGrupo().delete(eliminar);
			tm_afuera.get(pos_eliminar)
					.setMedia(
							Calcular_Media_Inst(tm_afuera.get(pos_eliminar)
									.getGrupo()));
			estruct.put(c, tm_afuera);
		} else {
			Eliminar_Grupo(c);
		}
	}

	// public Instances Eliminar_Elemento_Mas_Antiguo_Grupo(Instances grupo)
	// throws Exception {
	// if (grupo.numInstances() > 1) {
	// distanceFunction1.remove(grupo.instance(0));
	// grupo.delete(0);
	// } else {
	// Eliminar_Grupo(grupo.firstInstance().classValue());
	// }
	// return grupo;
	// }

	// public Instances Eliminar_Elemento_Mas_Antiguo_Menos_Util_Grupo(Instances
	// grupo) throws Exception {
	// int eliminar = 0;
	// double w = grupo.firstInstance().weight();
	//
	// if (grupo.numInstances() >1) {
	// for (int i = 1; i < grupo.numInstances(); i++) {
	// if (w > grupo.instance(i).weight()) {
	// w = grupo.instance(i).weight();
	// eliminar = i;
	// }
	// }
	// distanceFunction1.remove(grupo.instance(eliminar));
	// grupo.delete(eliminar);
	// } else {
	// Eliminar_Grupo(grupo.firstInstance().classValue());
	// }
	//
	// return grupo;
	//
	// }

	public void updateClassifier(Instance instance) {
		boolean missingValues = false;

		for (int i = 0; i < instance.numAttributes(); i++)
			if (instance.isMissing(i))
				missingValues = true;

		try {
			// if (!instance.hasMissingValue()) {
			if (!missingValues) {
				int r = (band == instance.classValue()) ? 1 : 0;
				lista_streaming.add(r);
				last100.add(instance);
				contador_total++;
				if (r == 1) {
					double w = (double) estruct
							.get(ubicacion_vec_mas_cercano.get(0))
							.get(ubicacion_vec_mas_cercano.get(1).intValue())
							.getGrupo()
							.instance(
									ubicacion_vec_mas_cercano.get(2).intValue())
							.weight() + 1.0;
					estruct.get(ubicacion_vec_mas_cercano.get(0))
							.get(ubicacion_vec_mas_cercano.get(1).intValue())
							.getGrupo()
							.instance(
									ubicacion_vec_mas_cercano.get(2).intValue())
							.setWeight(w);

					update_good(instance);
					cant_inst_bien_fols++;
					contadorC_aux++;
					// BienC.add(instance);
				} else {
					update_bad(instance);
					// MC.add(instance);
				}
				if (contador_total > 100) {
					last100.delete(0);
				}
				if (lista_streaming.size() == 100) {
					contadorC_aux--;
					lista_streaming.removeFirst();
					contadorMC_aux = (double) cant_inst_bien_fols
							/ contador_total;
					contador_streaming = (double) contadorC_aux / 100;

					if (((contadorMC_aux - contador_streaming) / contadorMC_aux) > 0.3) {
						update_with_concept_change();
						classifyInstance(instance);
					}
				}

			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public double classifyInstance(Instance instance) throws Exception {

		double[] dist = distributionForInstance(instance);
		if (dist == null) {
			throw new Exception("Null distribution predicted");
		}
		// switch (instance.classAttribute().type()) {
		// case Attribute.NOMINAL:
		// double max = 0;
		// int maxIndex = 0;
		//
		// for (int i = 0; i < dist.length; i++) {
		// if (dist[i] > max) {
		// maxIndex = i;
		// max = dist[i];
		// }
		// }
		// if (max > 0) {
		// band = maxIndex;
		// return maxIndex;
		// } else {
		// band = Instance.missingValue();
		// return Instance.missingValue();
		// }
		// case Attribute.NUMERIC:
		// band = dist[0];
		// return dist[0];
		// default:
		// band = Instance.missingValue();
		// return Instance.missingValue();
		// }
		if (instance.classAttribute().isNominal()) {
			double max = 0;
			int maxIndex = 0;

			for (int i = 0; i < dist.length; i++) {
				if (dist[i] > max) {
					maxIndex = i;
					max = dist[i];
				}
			}
			if (max > 0) {
				band = maxIndex;
				return maxIndex;
			} else {
				// band = Instance.missingValue();
				// return Instance.missingValue();
				band = -1;
				return band;
			}
		} else if (instance.classAttribute().isNumeric()) {
			band = dist[0];
			return dist[0];
		} else {
			// band = Instance.missingValue();
			// return Instance.missingValue();
			band = -1;
			return band;
		}
	}

	public void Devolver_neightbours(Instance instance) throws Exception {
		ubicacion_vec_mas_cercano = new LinkedList<Double>();
		TreeMap<Integer, Estructura> tm_afuera = new TreeMap<Integer, Estructura>();
		Estructura tm_adentro = new Estructura();
		Instances tm_grupo = new Instances(DS, 0);
		LinkedList<Instance> lista_inst = new LinkedList<Instance>();
		neightbours = new Instances(DS, 0);
		double d = 10000;
		double c = -1;
		Integer g = -1;
		Integer s = -1;
		double w = -1;
		for (int i = 0; i < lista_clases.size(); i++) {
			tm_afuera = estruct.get(lista_clases.get(i));
			int it = 0;
			for (int ii = 0; ii < tm_afuera.keySet().size(); ii++) {
				Integer[] array_it = tm_afuera.keySet().toArray(new Integer[0]);
				it = array_it[ii];
				tm_adentro = tm_afuera.get(it);
				tm_grupo = tm_adentro.getGrupo();
				// if(distanceFunction1.distance(instance,
				// tm_adentro.getMedia())<d){
				// lista_inst.add(tm_adentro.getMedia());
				// d = distanceFunction1.distance(instance,
				// tm_adentro.getMedia());
				//
				// }
				for (int iii = 0; iii < tm_grupo.numInstances(); iii++) {
					if (distanceFunction1.distance(instance,
							tm_grupo.instance(iii)) < d) {
						lista_inst.add(tm_grupo.instance(iii));
						d = distanceFunction1.distance(instance,
								tm_grupo.instance(iii));
						w = tm_grupo.instance(iii).weight();
						c = tm_grupo.instance(iii).classValue();
						g = it;
						s = iii;
					}
				}
			}
		}

		if (c != -1 && g != -1 && s != -1) {
			ubicacion_vec_mas_cercano.add(c);
			ubicacion_vec_mas_cercano.add(g.doubleValue());
			ubicacion_vec_mas_cercano.add(s.doubleValue());
		}

//		if (lista_inst.size() >= kNN) {
		if (lista_inst.size() >= kOption.getValue()) {
//			for (int i = lista_inst.size() - 1; i > lista_inst.size() - 1 - kNN; i--) {
			for (int i = lista_inst.size() - 1; i > lista_inst.size() - 1 - kOption.getValue(); i--) {
				neightbours.add(lista_inst.get(i));
			}
		} else {
			for (int i = 0; i < lista_inst.size(); i++) {
				neightbours.add(lista_inst.get(i));
			}
		}

	}

	public double[] distributionForInstance(Instance instance) {
		double[] kdist = new double[DS.numClasses()];

		try {
			/*
			 * Instances means = Devolver_DS_Final(); LinearNN lnn = new
			 * LinearNN(means); neightbours = null; neightbours =
			 * lnn.kNearestNeighbours(instance, kNN);
			 */
			Devolver_neightbours(instance);

//			for (int i = 0; i < Math.min(kNN, neightbours.numInstances()); i++) {
			for (int i = 0; i < Math.min(kOption.getValue(), neightbours.numInstances()); i++) {
				kdist[((int) neightbours.instance(i).classValue())]++;
			}

			for (int i = 0; i < DS.numClasses(); i++) {
//				kdist[i] /= kNN;
				kdist[i] /= kOption.getValue();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return kdist;
	}
	
	/**
	 * Extract from the method of the same name of weka.core.Attribute.
	 * 
	 * Returns the mean (mode) for a numeric (nominal) attribute as a
	 * floating-point value. Returns 0 if the attribute is neither nominal nor
	 * numeric. If all values are missing it returns zero.
	 * 
	 * @param attIndex
	 *            the attribute's index (index starts with 0)
	 * @return the mean or the mode
	 */
	private double meanOrMode(Instances insts, int attIndex) {

		double result, found;
		int[] counts;

		if (insts.attribute(attIndex).isNumeric()) {
			result = found = 0;
			for (int j = 0; j < insts.numInstances(); j++) {
				if (!insts.instance(j).isMissing(attIndex)) {
					found += insts.instance(j).weight();
					result += insts.instance(j).weight()
							* insts.instance(j).value(attIndex);
				}
			}
			if (found <= 0) {
				return 0;
			} else {
				return result / found;
			}
		} else if (insts.attribute(attIndex).isNominal()) {
			counts = new int[insts.attribute(attIndex).numValues()];
			for (int j = 0; j < insts.numInstances(); j++) {
				if (!insts.instance(j).isMissing(attIndex)) {
					counts[(int) insts.instance(j).value(attIndex)] += insts
							.instance(j).weight();
				}
			}
			return Utils.maxIndex(counts);
		} else {
			return 0;
		}
	}
	
	/**
	 * Updates the parameters that SimC needs in each prediction.
	 * 
	 * @param instance Instance to predict.
	 * @param dist Distribution for instance.
	 * @return Class predicted.
	 */
	public double classifyInstance(Instance instance, double[] dist) {
		if (instance.classAttribute().isNominal()) {
			double max = 0;
			int maxIndex = 0;

			for (int i = 0; i < dist.length; i++) {
				if (dist[i] > max) {
					maxIndex = i;
					max = dist[i];
				}
			}
			if (max > 0) {
				band = maxIndex;
				return maxIndex;
			} else {
				band = -1;
				return band;
			}
		} else if (instance.classAttribute().isNumeric()) {
			band = dist[0];
			return dist[0];
		} else {
			band = -1;
			return band;
		}
	}

	public String getPurposeString() {
		
		return "Similarity-based approach for data stream classification (SimC).";
	}

}
