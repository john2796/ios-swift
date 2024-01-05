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
 - timestamp: 1:07:18
 */

import SwiftUI

struct ContentView: View {

    @State private var isNight = false

    var body: some View {
        ZStack { // gradient background full screen
            BackgroundView(isNight: isNight)
            VStack {
                CityTextView(CityName: "Moreno Valley, CA")
                MainWeatherStatusView(imageName: isNight ? "moon.stars.fill" : "cloud.sun.fill", temperature: 76)
                
                HStack(spacing: 20) {
                    WeatherDayView(dayOfWeek: "TUE", imageName: "cloud.drizzle.fill", temperature: 76)
                    WeatherDayView(dayOfWeek: "WED", imageName: "cloud.rain.fill", temperature: 76)
                    WeatherDayView(dayOfWeek: "THU", imageName: "wind.snow", temperature: 76)
                    WeatherDayView(dayOfWeek: "FRI", imageName: "cloud.bolt.rain.fill", temperature: 76)
                    WeatherDayView(dayOfWeek: "SAT", imageName: "cloud.fog.fill", temperature: 76)
                }
                Spacer() // fill entire space
               
                
                Button {
                    isNight.toggle()
                } label: {
                    WeatherButton(title: "Change Day Time", textColor: .blue, backgroundColor: .white)
                }
                Spacer()
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

struct BackgroundView: View {
//    var topColor: Color
//    var bottomColor: Color
    var isNight: Bool
    var body: some View {
//        LinearGradient(gradient: Gradient(colors: [topColor, bottomColor]),             startPoint: .topLeading,
//                       endPoint: .bottomTrailing)
//        .edgesIgnoringSafeArea(.all)
        ContainerRelativeShape()
            .fill(isNight ? Color.black.gradient : Color.blue.gradient)
            .ignoresSafeArea()
    }
}

struct CityTextView: View {
    var CityName: String
    var body: some View {
        Text(CityName)
                .font(.system(size:32, weight: .medium, design: .default))
                .foregroundColor(.white)
                .padding()
    }
}

struct MainWeatherStatusView: View {
    var imageName: String
    var temperature: Int
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: imageName)
                .renderingMode(.original) // give color
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 180, height: 180)
            Text("\(temperature)°")
                .font(.system(size: 70, weight: .medium))
                .foregroundColor(.white)
        }
        .padding(.bottom, 40)
    }
}
