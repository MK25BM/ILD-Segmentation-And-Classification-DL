import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;


/**
*
* @author University & University Hospitals of Geneva (HUG) MedGIFT Group
*         Adrien Depeursinge adrien.depeursinge@hevs.ch
*
* 	   Copyright (C) 2012, MedGIFT group at University Hospitals of Geneva
*       
*      This program is free software; you can redistribute it and/or modify
*	   it under the terms of the GNU General Public License as published by
*      the Free Software Foundation; either version 2 of the License, or
*      (at your option) any later version.
*
*      This program is distributed in the hope that it will be useful,
*      but WITHOUT ANY WARRANTY; without even the implied warranty of
*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*      GNU General Public License for more details.
*
*      You should have received a copy of the GNU General Public License
*      along with this program; if not, write to the Free Software
*      Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA 
*			
*/

public class ROIReader {

	public ROIReader() {
		// nothing...
	}
	
	public ArrayList<Roi> readROIatzMPR(String pathROI, int zMPR) throws Exception {
		
		//read the *.txt file containing ROIs
		ArrayList<Roi> rois = new ArrayList<Roi>();
		try{
			
			InputStream ips = new FileInputStream(pathROI); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader reader = new BufferedReader(ipsr);
			String line = reader.readLine();
			
			String studyUID = "undefined";
			String seriesUID = "undefined";
			double xSpacing = -1;
			double ySpacing = -1;
			double zSpacing = -1;
			String label = "undefined";
			int sliceNumber = -1;
			
			while (line != null) {
				
				StringTokenizer tokenizer = new StringTokenizer(line, " ", false);
				if (tokenizer.countTokens() == 2) {

					String command = tokenizer.nextToken();
					String value = tokenizer.nextToken();
					
					if (command.equalsIgnoreCase("Study:")) {
						studyUID = value;
					} else if (command.equalsIgnoreCase("Series:")) {
						seriesUID = value;
					} else if (command.equalsIgnoreCase("SpacingX:")) {
						xSpacing = Double.parseDouble(value);
					} else if (command.equalsIgnoreCase("SpacingY:")) {
						ySpacing = Double.parseDouble(value);
					} else if (command.equalsIgnoreCase("SpacingZ:")) {
						zSpacing = Double.parseDouble(value);
					} else if (command.equalsIgnoreCase("label:")) {
						label = value;
					} else if (command.equalsIgnoreCase("slice_number:")) {
						sliceNumber = Integer.parseInt(value);
					} else if (command.equalsIgnoreCase("nb_points_on_contour:")) {
						int nbPointsOnContour = Integer.parseInt(value);
						double[] xValues = new double[nbPointsOnContour]; 
						double[] yValues = new double[nbPointsOnContour];
						
						for (int j=0; j<nbPointsOnContour; j++) {
							
							String linePt = reader.readLine();
							StringTokenizer tokenizerPt = new StringTokenizer(linePt, " ", false);
							String x = tokenizerPt.nextToken();
							String y = tokenizerPt.nextToken();
							
							xValues[j] = Double.parseDouble(x);
							yValues[j] = Double.parseDouble(y);
						}
						
						if (studyUID == "undefined") {
							System.err.println("Error: studyUID undefined");
							System.exit(1);
						}
						if (seriesUID == "undefined") {
							System.err.println("Error: seriesUID undefined");
							System.exit(1);
						}
						if (xSpacing == -1) {
							System.err.println("Error: xSpacing undefined");
			 				System.exit(1);
						}
						if (ySpacing == -1) {
							System.err.println("Error: ySpacing undefined");
							System.exit(1);
						}
						if (zSpacing == -1) {
							System.err.println("Error: zSpacing undefined");
							System.exit(1);
						}
						if (label == "undefined") {
							System.err.println("Error: label undefined");
							System.exit(1);
						}
						if (sliceNumber == -1) {
							System.err.println("Error: slice number undefined");
							System.exit(1);
						}
						
						// Storage of ROIs
						Roi roi = new Roi(xSpacing, ySpacing, zSpacing, label, sliceNumber, nbPointsOnContour);
						roi.setXValuesInMM(xValues);
						roi.setYValuesInMM(yValues);
						if (sliceNumber==zMPR+1)
							rois.add(roi);
					}
				}
				line = reader.readLine(); 
			}
		}
		catch (Exception e){
			System.err.println("invalid ROI *.txt file");
		}
		
		return rois;
	}
	
	public static ArrayList<Roi> readROI(String pathROI) throws Exception {
		
		//read the *.txt file containing ROIs
		ArrayList<Roi> rois = new ArrayList<Roi>();
		try{
			
			InputStream ips = new FileInputStream(pathROI); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader reader = new BufferedReader(ipsr);
			String line = reader.readLine();
			
			String studyUID = "undefined";
			String seriesUID = "undefined";
			double xSpacing = -1;
			double ySpacing = -1;
			double zSpacing = -1;
			String label = "undefined";
			int sliceNumber = -1;
			
			while (line != null) {
				
				StringTokenizer tokenizer = new StringTokenizer(line, " ", false);
				if (tokenizer.countTokens() == 2) {

					String command = tokenizer.nextToken();
					String value = tokenizer.nextToken();
					
					if (command.equalsIgnoreCase("Study:")) {
						studyUID = value;
					} else if (command.equalsIgnoreCase("Series:")) {
						seriesUID = value;
					} else if (command.equalsIgnoreCase("SpacingX:")) {
						xSpacing = Double.parseDouble(value);
					} else if (command.equalsIgnoreCase("SpacingY:")) {
						ySpacing = Double.parseDouble(value);
					} else if (command.equalsIgnoreCase("SpacingZ:")) {
						zSpacing = Double.parseDouble(value);
					} else if (command.equalsIgnoreCase("label:")) {
						label = value;
					} else if (command.equalsIgnoreCase("slice_number:")) {
						sliceNumber = Integer.parseInt(value);
					} else if (command.equalsIgnoreCase("nb_points_on_contour:")) {
						int nbPointsOnContour = Integer.parseInt(value);
						double[] xValues = new double[nbPointsOnContour]; 
						double[] yValues = new double[nbPointsOnContour];
						
						for (int j=0; j<nbPointsOnContour; j++) {
							
							String linePt = reader.readLine();
							StringTokenizer tokenizerPt = new StringTokenizer(linePt, " ", false);
							String x = tokenizerPt.nextToken();
							String y = tokenizerPt.nextToken();
							
							xValues[j] = Double.parseDouble(x);
							yValues[j] = Double.parseDouble(y);
						}
						
						if (studyUID == "undefined") {
							System.err.println("Error: studyUID undefined");
							System.exit(1);
						}
						if (seriesUID == "undefined") {
							System.err.println("Error: seriesUID undefined");
							System.exit(1);
						}
						if (xSpacing == -1) {
							System.err.println("Error: xSpacing undefined");
							System.exit(1);
						}
						if (ySpacing == -1) {
							System.err.println("Error: ySpacing undefined");
							System.exit(1);
						}
						if (zSpacing == -1) {
							System.err.println("Error: zSpacing undefined");
							System.exit(1);
						}
						if (label == "undefined") {
							System.err.println("Error: label undefined");
							System.exit(1);
						}
						if (sliceNumber == -1) {
							System.err.println("Error: slice number undefined");
							System.exit(1);
						}
						
						// Storage of ROIs
						Roi roi = new Roi(xSpacing, ySpacing, zSpacing, label, sliceNumber, nbPointsOnContour);
						roi.setXValuesInMM(xValues);
						roi.setYValuesInMM(yValues);
						roi.setStudyUID(studyUID);
						roi.setSeriesUID(seriesUID);
						rois.add(roi);
					}
				}
				line = reader.readLine(); 
			}
		}
		catch (Exception e){
			System.err.println("invalid ROI *.txt file");
		}
		
		return rois;
	}
}