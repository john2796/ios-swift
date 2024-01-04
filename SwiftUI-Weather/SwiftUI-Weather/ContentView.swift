//
//  ContentView.swift
//  SwiftUI-Weather
//
//  Created by john benedict miranda on 1/3/24.
//

/**
 Modifiers orders matter
 
 https://www.youtube.com/watch?v=b1oC7sLIgpI
 - timestamp: 46:50
 */

import SwiftUI

struct ContentView: View {
    var body: some View {
        ZStack { // gradient background full screen
            LinearGradient(gradient: Gradient(colors: [.blue, .white]),             startPoint: .topLeading,
                           endPoint: .bottomTrailing)
                .edgesIgnoringSafeArea(.all)
            VStack {
                Text("Moreno Valley, CA")
                    .font(.system(size:32, weight: .medium, design: .default))
                    .foregroundColor(.white)
                    .padding()
                VStack(spacing: 8) {
                    Image(systemName: "cloud.sun.fill")
                        .renderingMode(.original) // give color
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 180, height: 180)
                    Text("76°")
                        .font(.system(size: 70, weight: .medium))
                        .foregroundColor(.white)
                }
                Spacer()
                
                HStack(spacing: 20) {
                    WeatherDayView(dayOfWeek: "TUE", imageName: "cloud.drizzle.fill", temperature: 76)
                    WeatherDayView(dayOfWeek: "WED", imageName: "cloud.rain.fill", temperature: 76)
                    WeatherDayView(dayOfWeek: "THU", imageName: "wind.snow", temperature: 76)
                    WeatherDayView(dayOfWeek: "FRI", imageName: "cloud.bolt.rain.fill", temperature: 76)
                    WeatherDayView(dayOfWeek: "SAT", imageName: "cloud.fog.fill", temperature: 76)
                }
                Spacer() // fill entire space
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}


// reusable weather day view
struct WeatherDayView: View {
    var dayOfWeek: String
    var imageName: String
    var temperature: Int
    
    var body: some View {
        VStack {
            Text(dayOfWeek)
                .font(.system(size: 16, weight: .medium, design: .default))
                .foregroundColor(.white)
            Image(systemName: imageName)
                .renderingMode(.original)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 40, height: 40)
            Text("\(temperature)°")
                .font(.system(size: 28, weight: .medium))
                .foregroundColor(.white)
        }
    }
}
